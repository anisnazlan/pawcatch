"""Build the training dataset from several downloaded sources.

It copies photos into the 6 class folders train.py expects, and on the way it:
  - reads YOLO detection datasets (data.yaml + images + labels) and cuts out each boxed animal
  - converts WEBP/BMP/GIF/TIFF to JPG and skips files that aren't images or are broken
  - skips tiny images (under 100 px)
  - removes duplicates, including resized copies of the same photo
  - shrinks large photos (longest side 640 px) so the zip uploads fast
  - balances the classes by keeping at most --max photos per class, mixed evenly
    across sources (so "other" gets a bit of every animal, not mostly one)
  - can download iNaturalist photos straight from an export CSV

Example with the WildWatch datasets (on Windows Command Prompt, end lines with ^ instead):

    python training/prepare_dataset.py \
        --yolo downloads/wildlife74 \
        --add dog   downloads/catdogsnake/dogs \
        --add snake downloads/catdogsnake/snakes \
        --inat monitor_lizard downloads/inat_biawak.csv \
        --inat other downloads/inat_cicak.csv \
        --max 250 --zip

--yolo uses YOLO_MAP below to decide which dataset class goes into which of our classes.
Add or change one with --yolo-map NAME=CLASS (e.g. --yolo-map lizard=other), or leave
a class out with --yolo-map NAME=skip.

Other sources (one line per source; the same class can appear several times):

    python training/prepare_dataset.py \
        --add other     downloads/animals90/animals/cat \
        --add other     downloads/animals90/animals/squirrel \
        --add wild_boar downloads/animals90/animals/boar \
        --add dog       downloads/animals10/raw-img/cane \
        --add snake     downloads/snakes \
        --inat monkey    downloads/inat_macaque.csv \
        --inat wild_boar downloads/inat_boar.csv \
        --max 300 --zip

Output: dataset/monkey, dog, snake, wild_boar, monitor_lizard, other
(and dataset.zip with --zip). A summary at the end shows how many photos each class got.
"""
import argparse
import csv
import io
import random
import shutil
import sys
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

CLASSES = ["monkey", "dog", "snake", "wild_boar", "monitor_lizard", "other"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"}
MIN_SIDE = 100
MAX_SIDE = 640
SIMILAR_BITS = 6  # out of 64; higher = stricter duplicate removal
BOX_PADDING = 0.10  # extra space kept around each YOLO box (10% of its size per side)

# YOLO dataset class name -> our class. Names are compared in lower case, with spaces
# and dashes treated as "_". Classes not listed here are ignored.
YOLO_MAP = {
    "monkey": "monkey",
    "dog": "dog",
    "wild_boar": "wild_boar",
    "rattlesnake": "snake",
    "northern_pacific_rattlesnake": "snake",
    "western_diamondback_rattlesnake": "snake",
    "cat": "other", "squirrel": "other", "crow": "other", "chicken": "other",
    "sparrow": "other", "mongoose": "other", "rabbit": "other", "duck": "other",
    # "wild_dog" is left out on purpose: it looks like a stray dog but is a different case.
}

# With --yolo-rest-other, every other YOLO class goes into "other", EXCEPT animals that
# look like one of our 5 (they would confuse the AI). Matched as parts of the class name.
LOOKALIKE_WORDS = ["dog", "wolf", "fox", "jackal", "coyote", "hyena", "dingo",
                   "pig", "hog", "boar", "warthog",
                   "lizard", "iguana", "gecko", "monitor", "komodo", "crocodile", "alligator", "gharial",
                   "snake", "python", "cobra", "viper", "rattle", "mamba", "adder", "boa",
                   "monkey", "ape", "gorilla", "chimp", "orangutan", "baboon", "macaque", "lemur", "gibbon",
                   "ipynb", "checkpoint"]
Image.MAX_IMAGE_PIXELS = 60_000_000


def dhash(img: Image.Image, size: int = 8) -> int:
    """Perceptual hash: the same photo gives the same hash even after resizing."""
    g = img.convert("L").resize((size + 1, size), Image.LANCZOS)
    px = g.tobytes()  # one brightness value per pixel
    bits = 0
    for row in range(size):
        for col in range(size):
            left = px[row * (size + 1) + col]
            right = px[row * (size + 1) + col + 1]
            bits = (bits << 1) | (left > right)
    return bits


def norm_name(name: str) -> str:
    return "_".join(str(name).strip().lower().replace("-", " ").replace("_", " ").split())


def read_yolo_names(yaml_path: Path) -> list:
    """Class names from a YOLO data.yaml (list, inline list or numbered form), without needing PyYAML."""
    lines = yaml_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    for i, line in enumerate(lines):
        if not line.lstrip().startswith("names:") or line.startswith((" ", "\t")):
            continue
        rest = line.split(":", 1)[1].strip()
        if rest.startswith("["):  # names: ['cat', 'dog']
            body = rest
            j = i
            while "]" not in body and j + 1 < len(lines):
                j += 1
                body += lines[j]
            body = body[body.index("[") + 1: body.index("]")]
            return [x.strip().strip("'\"") for x in body.split(",") if x.strip()]
        names, numbered = [], {}
        for nxt in lines[i + 1:]:
            t = nxt.strip()
            if not t or t.startswith("#"):
                continue
            if not nxt.startswith((" ", "\t", "-")):
                break
            if t.startswith("- "):          # - cat
                names.append(t[2:].strip().strip("'\""))
            elif ":" in t:                   # 0: cat
                k, v = t.split(":", 1)
                if k.strip().isdigit():
                    numbered[int(k)] = v.strip().strip("'\"")
        if numbered:
            return [numbered.get(k, f"class_{k}") for k in range(max(numbered) + 1)]
        return names
    return []


def yolo_label_for(img: Path):
    """images/train/x.jpg -> labels/train/x.txt (the usual YOLO layout)."""
    parts = list(img.parts)
    for k in range(len(parts) - 1, -1, -1):
        if parts[k].lower() == "images":
            parts[k] = "labels" if parts[k] == "images" else "Labels"
            cand = Path(*parts).with_suffix(".txt")
            if cand.exists():
                return cand
    cand = img.with_suffix(".txt")
    return cand if cand.exists() else None


def yolo_candidates(folder: Path, mapping: dict, stats, rest_other=False):
    """Yield (our class, source name, crop) for every mapped box in a YOLO dataset."""
    yamls = sorted(folder.rglob("data.yaml"), key=lambda p: len(p.parts))
    if not yamls:
        print(f"  ! no data.yaml found under {folder}")
        return
    names = read_yolo_names(yamls[0])
    if not names:
        print(f"  ! could not read the class names in {yamls[0]}")
        return
    print(f"  YOLO dataset {folder}: {len(names)} classes in {yamls[0].name}")
    targets = {i: mapping.get(norm_name(n)) for i, n in enumerate(names)}
    if rest_other:
        added, kept_out = [], []
        for i, n in enumerate(names):
            if targets.get(i):
                continue
            if any(w in norm_name(n) for w in LOOKALIKE_WORDS):
                kept_out.append(n)
            else:
                targets[i] = "other"; added.append(n)
        print(f"    added to other ({len(added)}): {', '.join(added)}")
        print(f"    kept out as look-alikes ({len(kept_out)}): {', '.join(kept_out)}")
    targets = {i: c for i, c in targets.items() if c and c != "skip"}
    used = sorted({names[i] for i in targets}, key=str.lower)
    print(f"    using {len(used)}: {', '.join(used)}")
    missing = [k for k, v in mapping.items() if v != "skip" and k not in {norm_name(n) for n in names}]
    if missing:
        print(f"    not in this dataset: {', '.join(missing)}")
    images = [f for f in folder.rglob("*") if f.is_file() and f.suffix.lower() in IMAGE_EXTS]
    no_label = 0
    for img in images:
        lab = yolo_label_for(img)
        if lab is None:
            no_label += 1
            continue
        for row in lab.read_text(errors="ignore").splitlines():
            v = row.split()
            if len(v) < 5:
                continue
            try:
                cid = int(float(v[0]))
                nums = [float(x) for x in v[1:]]
            except ValueError:
                continue
            if cid not in targets:
                continue
            if len(nums) == 4:               # box: centre x, centre y, width, height
                cx, cy, w, h = nums
                box = (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
            else:                            # polygon: x1 y1 x2 y2 ...
                xs, ys = nums[0::2], nums[1::2]
                box = (min(xs), min(ys), max(xs), max(ys))
            yield targets[cid], names[cid], ("crop", img, box)
    if no_label:
        print(f"    {no_label} images had no label file (skipped)")


def load_image(data_or_path):
    """Open an image safely. Returns an RGB image or None if unusable."""
    if isinstance(data_or_path, tuple) and data_or_path[0] == "crop":
        return load_crop(data_or_path[1], data_or_path[2])
    return load_image_full(data_or_path)


def load_image_full(data_or_path):
    try:
        src = io.BytesIO(data_or_path) if isinstance(data_or_path, bytes) else data_or_path
        with Image.open(src) as im:
            im.seek(0)  # first frame of GIFs
            im = ImageOps.exif_transpose(im)
            im = im.convert("RGB")
            im.load()
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
        return None
    if min(im.size) < MIN_SIDE:
        return None
    return im


def load_crop(img_path: Path, box):
    """Cut one boxed animal out of a photo, with a little padding around it."""
    im = load_image_full(img_path)
    if im is None:
        return None
    W, H = im.size
    x1, y1, x2, y2 = box
    bw, bh = (x2 - x1), (y2 - y1)
    x1, x2 = x1 - bw * BOX_PADDING, x2 + bw * BOX_PADDING
    y1, y2 = y1 - bh * BOX_PADDING, y2 + bh * BOX_PADDING
    l, t = max(0, int(x1 * W)), max(0, int(y1 * H))
    r, b = min(W, int(x2 * W)), min(H, int(y2 * H))
    if r - l < MIN_SIDE or b - t < MIN_SIDE:
        return None
    return im.crop((l, t, r, b))


def download_inat(csv_path: Path, limit: int, delay: float):
    """Yield image bytes from an iNaturalist observations export (image_url column)."""
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    urls = [r.get("image_url", "").strip() for r in rows]
    urls = [u for u in urls if u.startswith("http")]
    if not urls:
        print(f"  ! {csv_path.name}: no image_url column or no URLs found")
        return
    random.shuffle(urls)
    print(f"  downloading up to {min(limit, len(urls))} photos from {csv_path.name} ({len(urls)} listed) ...")
    fetched = 0
    for url in urls:
        if fetched >= limit:
            break
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "WildWatch-student-project/1.0"})
            with urllib.request.urlopen(req, timeout=20) as r:
                data = r.read()
        except Exception:
            continue
        fetched += 1
        yield url, data
        time.sleep(delay)  # be polite to iNaturalist's servers


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--add", nargs=2, action="append", default=[], metavar=("CLASS", "FOLDER"),
                   help="add every image under FOLDER (subfolders included) to CLASS")
    p.add_argument("--yolo", action="append", default=[], metavar="FOLDER",
                   help="YOLO dataset folder (with data.yaml); boxes are cut out and mapped with YOLO_MAP")
    p.add_argument("--yolo-rest-other", action="store_true",
                   help="put every other YOLO class into 'other' (look-alikes of our 5 animals are kept out)")
    p.add_argument("--yolo-map", action="append", default=[], metavar="NAME=CLASS",
                   help="change the class mapping for --yolo, e.g. lizard=other or wild_dog=skip")
    p.add_argument("--inat", nargs=2, action="append", default=[], metavar=("CLASS", "CSV"),
                   help="download photos listed in an iNaturalist export CSV into CLASS")
    p.add_argument("--out", default="dataset", help="output folder (default: dataset)")
    p.add_argument("--max", type=int, default=300, help="max photos per class (default: 300)")
    p.add_argument("--max-other", type=int, default=None,
                   help="max photos for 'other' (default: same as --max). Bigger = more variety of 'not our animals'")
    p.add_argument("--min-warn", type=int, default=150, help="warn if a class has fewer (default: 150)")
    p.add_argument("--inat-limit", type=int, default=400, help="max downloads per CSV (default: 400)")
    p.add_argument("--inat-delay", type=float, default=0.3, help="seconds between downloads")
    p.add_argument("--zip", action="store_true", help="also create <out>.zip for uploading to Colab")
    p.add_argument("--zip-only", action="store_true",
                   help="don't rebuild: just zip the existing dataset folder (use after deleting bad photos)")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()

    random.seed(a.seed)
    cap = lambda c: (a.max_other or a.max) if c == "other" else a.max
    if a.zip_only:
        out = Path(a.out)
        if not out.is_dir():
            sys.exit(f"No {out}/ folder yet. Run the full command first.")
        print("Photos per class:")
        for c in CLASSES:
            n = len(list((out / c).glob("*.jpg"))) if (out / c).is_dir() else 0
            print(f"  {c:<15} {n:>4}" + ("  <- too few, add more photos" if n < a.min_warn else ""))
        shutil.make_archive(str(out), "zip", root_dir=out.parent, base_dir=out.name)
        print(f"\nCreated {out}.zip - upload this to the Colab notebook.")
        return
    for cls, _ in a.add + a.inat:
        if cls not in CLASSES:
            sys.exit(f"Unknown class '{cls}'. Use one of: {', '.join(CLASSES)}")
    mapping = dict(YOLO_MAP)
    for m in a.yolo_map:
        if "=" not in m:
            sys.exit(f"--yolo-map needs NAME=CLASS, got '{m}'")
        k, v = m.split("=", 1)
        if v not in CLASSES and v != "skip":
            sys.exit(f"--yolo-map {m}: '{v}' must be one of {', '.join(CLASSES)} or skip")
        mapping[norm_name(k)] = v
    if not a.add and not a.inat and not a.yolo:
        sys.exit("Nothing to do. Add sources with --yolo FOLDER, --add CLASS FOLDER or --inat CLASS CSV (see --help).")

    out = Path(a.out)
    if out.exists():
        shutil.rmtree(out)
    for c in CLASSES:
        (out / c).mkdir(parents=True)

    # Collect candidates per class: (source label, path-or-bytes)
    candidates = defaultdict(list)
    for cls, folder in a.add:
        folder = Path(folder)
        if not folder.is_dir():
            print(f"  ! folder not found, skipped: {folder}")
            continue
        files = [f for f in folder.rglob("*") if f.is_file() and f.suffix.lower() in IMAGE_EXTS]
        print(f"  {cls:<9} <- {len(files):>5} files from {folder}")
        candidates[cls] += [(str(folder), f) for f in files]

    # iNaturalist: download once into a folder next to the CSV, then reuse it on later runs.
    for cls, csv_path in a.inat:
        csv_path = Path(csv_path)
        if not csv_path.is_file():
            print(f"  ! CSV not found, skipped: {csv_path}")
            continue
        cache = csv_path.with_name(csv_path.stem + "_photos")
        files = sorted(cache.glob("*.jpg")) if cache.is_dir() else []
        if files:
            print(f"  {cls:<9} <- {len(files):>5} photos already downloaded in {cache}")
        else:
            cache.mkdir(exist_ok=True)
            for i, (_, data) in enumerate(download_inat(csv_path, min(a.inat_limit, cap(cls) + 20), a.inat_delay)):
                (cache / f"inat_{i:04d}.jpg").write_bytes(data)
            files = sorted(cache.glob("*.jpg"))
            print(f"  {cls:<9} <- {len(files):>5} photos downloaded into {cache}")
        candidates[cls] += [(csv_path.stem, f) for f in files]

    stats = defaultdict(lambda: defaultdict(int))
    for folder in a.yolo:
        folder = Path(folder)
        if not folder.is_dir():
            print(f"  ! folder not found, skipped: {folder}")
            continue
        found = defaultdict(int)
        for cls, src, item in yolo_candidates(folder, mapping, stats, a.yolo_rest_other):
            candidates[cls].append((src, item))
            found[cls] += 1
        for cls in CLASSES:
            if found[cls]:
                print(f"  {cls:<9} <- {found[cls]:>5} boxes from {folder}")

    by_source = defaultdict(lambda: defaultdict(int))  # photos kept per class per source
    seen = {}                 # photo hash -> (class, saved file), across all classes
    kept = defaultdict(int)   # photos saved per class

    def find_similar(h):
        """Near-identical photos (re-saved, resized, converted) differ by only a few bits."""
        for other in seen:
            if (h ^ other).bit_count() <= SIMILAR_BITS:
                return other
        return None

    def consider(cls, item) -> bool:
        """Check one photo and save it. Returns False once the class is full."""
        if kept[cls] >= cap(cls):
            return False
        im = load_image(item)
        if im is None:
            stats[cls]["broken or too small"] += 1
            return True
        h = dhash(im)
        match = find_similar(h)
        if match is not None:
            other_cls, other_file = seen[match]
            if other_cls == cls:
                stats[cls]["duplicate"] += 1
            elif other_file is not None:  # same photo labelled as two animals: drop both
                other_file.unlink(missing_ok=True)
                kept[other_cls] -= 1
                stats[cls]["in two classes (removed)"] += 1
                seen[match] = (other_cls, None)
            return True
        kept[cls] += 1
        im.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
        path = out / cls / f"{cls}_{len(seen):05d}.jpg"
        im.save(path, "JPEG", quality=90)
        seen[h] = (cls, path)
        return True

    def mix(items):
        """Shuffle, then take one from each source in turn, so every source gets a fair share."""
        groups = defaultdict(list)
        for src, item in items:
            groups[norm_name(src)].append((src, item))
        lists = list(groups.values())
        for g in lists:
            random.shuffle(g)
        random.shuffle(lists)
        out = []
        while any(lists):
            for g in lists:
                if g:
                    out.append(g.pop())
        return out

    for cls in list(candidates):
        items = mix(candidates[cls])
        for i, (src, item) in enumerate(items):
            before = kept[cls]
            if not consider(cls, item):
                stats[cls]["extra, class already full"] += len(items) - i
                break
            if kept[cls] > before:
                by_source[cls][src] += 1
    # Rename to a clean sequence: monkey_0001.jpg, monkey_0002.jpg ...
    summary = []
    for cls in CLASSES:
        files = sorted((out / cls).glob("*.jpg"))
        for i, f in enumerate(files, 1):
            f.rename(out / cls / f"{cls}_{i:04d}.tmp")
        for f in sorted((out / cls).glob("*.tmp")):
            f.rename(f.with_suffix(".jpg"))
        summary.append((cls, len(files)))

    print("\nSummary")
    print("-" * 48)
    for cls, n in summary:
        flag = "  <- too few, add more photos" if n < a.min_warn else ""
        print(f"  {cls:<15} {n:>4} photos{flag}")
        if len(by_source[cls]) > 1:
            mix_txt = ", ".join(f"{Path(s).name} {k}" for s, k in sorted(by_source[cls].items(), key=lambda x: -x[1]))
            print(f"      from: {mix_txt}")
        for reason, k in stats[cls].items():
            print(f"      skipped {k} ({reason})")
    counts = [n for c, n in summary if not (a.max_other and c == "other")]
    if counts and min(counts) and max(counts) / min(counts) > 2:
        print("\n  Note: the biggest class has over twice the photos of the smallest.")
        print("  Lower --max to the smallest count for a fairer model.")

    if a.zip:
        shutil.make_archive(str(out), "zip", root_dir=out.parent, base_dir=out.name)
        print(f"\nCreated {out}.zip - upload this to the Colab notebook.")
    print(f"\nDone. Open the {out}/ folders and delete any photo that is clearly wrong,")
    print("then run:  python training/prepare_dataset.py --zip-only")
    print("(Running the full command again would rebuild the folders and bring those photos back.)")


if __name__ == "__main__":
    main()