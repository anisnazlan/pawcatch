"""Build the training dataset from several downloaded sources.

It copies photos into the 5 class folders train.py expects, and on the way it:
  - converts WEBP/BMP/GIF/TIFF to JPG and skips files that aren't images or are broken
  - skips tiny images (under 100 px)
  - removes duplicates, including resized copies of the same photo
  - shrinks large photos (longest side 640 px) so the zip uploads fast
  - balances the classes by keeping at most --max photos per class (picked at random)
  - can download iNaturalist photos straight from an export CSV

Example (one line per source; the same class can appear several times):

    python training/prepare_dataset.py \
        --add other     downloads/animals90/animals/cat \
        --add other     downloads/animals90/animals/squirrel \
        --add wild_boar downloads/animals90/animals/boar \
        --add dog       downloads/animals10/raw-img/cane \
        --add snake     downloads/snakes \
        --inat monkey    downloads/inat_macaque.csv \
        --inat wild_boar downloads/inat_boar.csv \
        --max 300 --zip

Output: dataset/monkey, dataset/dog, dataset/snake, dataset/wild_boar, dataset/other
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

CLASSES = ["monkey", "dog", "snake", "wild_boar", "other"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"}
MIN_SIDE = 100
MAX_SIDE = 640
SIMILAR_BITS = 6  # out of 64; higher = stricter duplicate removal
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


def load_image(data_or_path):
    """Open an image safely. Returns an RGB image or None if unusable."""
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
    p.add_argument("--inat", nargs=2, action="append", default=[], metavar=("CLASS", "CSV"),
                   help="download photos listed in an iNaturalist export CSV into CLASS")
    p.add_argument("--out", default="dataset", help="output folder (default: dataset)")
    p.add_argument("--max", type=int, default=300, help="max photos per class (default: 300)")
    p.add_argument("--min-warn", type=int, default=150, help="warn if a class has fewer (default: 150)")
    p.add_argument("--inat-limit", type=int, default=400, help="max downloads per CSV (default: 400)")
    p.add_argument("--inat-delay", type=float, default=0.3, help="seconds between downloads")
    p.add_argument("--zip", action="store_true", help="also create <out>.zip for uploading to Colab")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()

    random.seed(a.seed)
    for cls, _ in a.add + a.inat:
        if cls not in CLASSES:
            sys.exit(f"Unknown class '{cls}'. Use one of: {', '.join(CLASSES)}")
    if not a.add and not a.inat:
        sys.exit("Nothing to do. Add sources with --add CLASS FOLDER or --inat CLASS CSV (see --help).")

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

    stats = defaultdict(lambda: defaultdict(int))
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
        if kept[cls] >= a.max:
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

    for cls, items in candidates.items():
        random.shuffle(items)
        for i, (_, item) in enumerate(items):
            if not consider(cls, item):
                stats[cls]["extra, class already full"] += len(items) - i
                break
    for cls, csv_path in a.inat:
        csv_path = Path(csv_path)
        if not csv_path.is_file():
            print(f"  ! CSV not found, skipped: {csv_path}")
            continue
        if kept[cls] >= a.max:
            print(f"  {cls} already has {a.max} photos, skipping {csv_path.name}")
            continue
        for _, data in download_inat(csv_path, min(a.inat_limit, a.max - kept[cls] + 20), a.inat_delay):
            if not consider(cls, data):
                break

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
        print(f"  {cls:<10} {n:>4} photos{flag}")
        for reason, k in stats[cls].items():
            print(f"      skipped {k} ({reason})")
    counts = [n for _, n in summary]
    if counts and min(counts) and max(counts) / min(counts) > 2:
        print("\n  Note: the biggest class has over twice the photos of the smallest.")
        print("  Lower --max to the smallest count for a fairer model.")

    if a.zip:
        shutil.make_archive(str(out), "zip", root_dir=out.parent, base_dir=out.name)
        print(f"\nCreated {out}.zip - upload this to the Colab notebook.")
    print(f"\nDone. Open the {out}/ folders and delete any photo that is clearly wrong.")


if __name__ == "__main__":
    main()
