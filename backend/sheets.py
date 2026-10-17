"""Optional Google Sheet copy of live reports.

When SHEETS_URL and SHEETS_SECRET are set (in Render's Environment settings), every
live report is copied to a Google Sheet, status changes are copied too, and when the
server restarts the reports are loaded back from the sheet. Without them, nothing here runs.

The sheet side is google_sheets/Code.gs (a Google Apps Script web app).
"""
import json
import os
import queue
import threading
import time
import urllib.parse
import urllib.request
from datetime import datetime

SHEETS_URL = os.getenv("SHEETS_URL", "").strip()
SHEETS_SECRET = os.getenv("SHEETS_SECRET", "").strip()
ENABLED = bool(SHEETS_URL and SHEETS_SECRET)

FIELDS = ["id", "created_at", "lat", "lng", "animal", "incident", "urgency", "agency",
          "status", "confidence", "ai_mode", "photo", "note", "resolved_at"]

_queue: "queue.Queue[dict]" = queue.Queue()
_worker = None
_lock = threading.Lock()


def _request(payload=None, timeout=30):
    """POST payload (or GET when payload is None). Apps Script answers through a redirect,
    which urllib follows automatically."""
    if payload is None:
        url = SHEETS_URL + ("&" if "?" in SHEETS_URL else "?") + urllib.parse.urlencode({"secret": SHEETS_SECRET})
        req = urllib.request.Request(url)
    else:
        body = json.dumps({**payload, "secret": SHEETS_SECRET}).encode()
        req = urllib.request.Request(SHEETS_URL, data=body, headers={"Content-Type": "text/plain;charset=utf-8"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        result = json.loads(r.read().decode() or "{}")
    if not result.get("ok"):
        raise RuntimeError(result.get("error", "the sheet did not accept the request"))
    return result


def _run():
    # One worker sends changes in order, so a status update never arrives before its report.
    while True:
        payload = _queue.get()
        for attempt in range(4):
            try:
                _request(payload)
                break
            except Exception as e:
                if attempt == 3:
                    print(f"[sheets] gave up sending {payload.get('action')} for report "
                          f"{payload.get('id') or payload.get('report', {}).get('id')}: {e}", flush=True)
                else:
                    time.sleep(2 * (attempt + 1))
        _queue.task_done()


def _send(payload):
    global _worker
    if not ENABLED:
        return
    with _lock:
        if _worker is None:
            _worker = threading.Thread(target=_run, daemon=True)
            _worker.start()
    _queue.put(payload)


def add_report(row: dict):
    """Copy a new live report to the sheet (in the background, the resident doesn't wait)."""
    _send({"action": "add", "report": {k: row.get(k) for k in FIELDS}})


def update_status(rid: int, status: str, resolved_at):
    _send({"action": "status", "id": rid, "status": status, "resolved_at": resolved_at})


def _local_time(value):
    """Sheets may hand dates back in UTC ('...Z'); turn them into local time like the app uses."""
    if not value:
        return None
    s = str(value)
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return s
    if dt.tzinfo is not None:
        dt = dt.astimezone().replace(tzinfo=None)
    return dt.isoformat(timespec="seconds")


def load_reports():
    """Read every report back from the sheet. Returns [] if the sheet can't be reached."""
    if not ENABLED:
        return []
    try:
        rows = _request(None, timeout=45).get("rows", [])
    except Exception as e:
        print(f"[sheets] could not load reports from the sheet: {e}", flush=True)
        return []
    out = []
    for r in rows:
        try:
            out.append({
                "id": int(float(r["id"])),
                "created_at": _local_time(r.get("created_at")),
                "lat": float(r["lat"]), "lng": float(r["lng"]),
                "animal": str(r["animal"]), "incident": str(r["incident"]),
                "urgency": str(r["urgency"]), "agency": str(r["agency"]),
                "status": str(r.get("status") or "assigned"),
                "confidence": float(r["confidence"]) if r.get("confidence") not in (None, "") else None,
                "ai_mode": r.get("ai_mode") or None,
                "note": r.get("note") or None,
                "resolved_at": _local_time(r.get("resolved_at")),
            })
        except (KeyError, TypeError, ValueError):
            continue  # skip a row someone edited into an unreadable state
    return [r for r in out if r["created_at"]]
