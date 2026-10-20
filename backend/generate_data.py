"""Generates SIMULATED sighting reports around City of Elmina for the dashboard demo.

Patterns are built in on purpose so the hotspot map tells a clear story.
All rows are flagged simulated=1. Coordinates are approximate; adjust HOTSPOTS
to match real streets, bins and forest edges once you check the map.

Run from project root:
    python -m backend.generate_data --reset      replace simulated rows
    python -m backend.generate_data --if-empty   only seed when there is no simulated data yet
                                                 (used on Render, so restarts keep the same data)
"""
import argparse
import random
from datetime import datetime, timedelta

from . import rules
from .db import init_db, query, query_many

# (name, lat, lng, spread, weight, animal mix, incident mix, peak hours)
HOTSPOTS = [
    ("Forest edge, Elmina West", 3.1835, 101.4785, 0.0012, 0.25,
     {"monkey": 0.6, "wild_boar": 0.25, "snake": 0.15},
     {"sighting": 0.4, "bins": 0.3, "entered_home": 0.2, "damage": 0.1}, [6, 7, 8, 18]),
    ("Bin area, Elmina East precinct", 3.1765, 101.4960, 0.0008, 0.22,
     {"monkey": 0.7, "dog": 0.3},
     {"bins": 0.7, "sighting": 0.2, "damage": 0.1}, [6, 7, 8]),
    ("Playground and park", 3.1710, 101.5035, 0.0009, 0.18,
     {"dog": 0.85, "monkey": 0.15},
     {"sighting": 0.55, "attack": 0.15, "entered_home": 0.3}, [17, 18, 19, 20]),
    ("Drain and grass strip", 3.1880, 101.4920, 0.0010, 0.12,
     {"snake": 0.55, "monitor_lizard": 0.3, "other": 0.15},
     {"sighting": 0.5, "entered_home": 0.5}, [10, 11, 15, 21]),
    ("Lake and monsoon drain", 3.1745, 101.4880, 0.0009, 0.08,
     {"monitor_lizard": 0.7, "snake": 0.2, "other": 0.1},
     {"sighting": 0.5, "entered_home": 0.3, "damage": 0.2}, [9, 10, 11, 16, 17]),
    ("Commercial area", 3.1790, 101.5080, 0.0011, 0.10,
     {"dog": 0.7, "monkey": 0.3},
     {"bins": 0.6, "sighting": 0.4}, [21, 22, 23, 6]),
]
BACKGROUND = (3.1800, 101.4930, 0.009)  # scattered reports across the township
STATUS_WEIGHTS = {"resolved": 0.7, "in_progress": 0.12, "assigned": 0.13, "received": 0.05}


def pick(weights: dict) -> str:
    return random.choices(list(weights), weights=list(weights.values()))[0]


def main(n: int, reset: bool, if_empty: bool, seed: int):
    random.seed(seed)
    init_db()
    if if_empty and query("SELECT COUNT(*) AS c FROM reports WHERE simulated=1", fetch="one")["c"]:
        print("Simulated data already present, skipping.")
        return
    now = datetime.now()
    rows = []
    for _ in range(n):
        if random.random() < 0.85:
            _, lat0, lng0, spread, _, animals, incidents, hours = random.choices(
                HOTSPOTS, weights=[h[4] for h in HOTSPOTS])[0]
            hour = random.choice(hours) if random.random() < 0.75 else random.randint(0, 23)
        else:
            lat0, lng0, spread = BACKGROUND
            animals = {"monkey": 0.3, "dog": 0.33, "snake": 0.13, "wild_boar": 0.05, "monitor_lizard": 0.07, "other": 0.12}
            incidents = {"sighting": 0.6, "bins": 0.2, "entered_home": 0.15, "attack": 0.05}
            hour = random.randint(0, 23)
        lat = random.gauss(lat0, spread)
        lng = random.gauss(lng0, spread)
        animal, incident = pick(animals), pick(incidents)
        # Reports increase over recent months to show a growing problem.
        days_ago = int(abs(random.triangular(0, 180, 0)))
        created = (now - timedelta(days=days_ago)).replace(
            hour=hour, minute=random.randint(0, 59), second=0, microsecond=0)
        if created > now:
            created -= timedelta(days=1)
        status = pick(STATUS_WEIGHTS) if days_ago < 14 else "resolved"
        resolved_at = None
        if status == "resolved":
            # Realistic sample times per agency (hours): Bomba is fast, the council is slowest.
            low, high = {"emergency": (1, 6), "township": (4, 30), "perhilitan": (8, 40), "council": (20, 70)}[
                rules.agency_for(animal, incident)]
            hours_to_fix = random.uniform(low, high)
            resolved_at = (created + timedelta(hours=hours_to_fix)).isoformat(timespec="seconds")
        rows.append((
            created.isoformat(timespec="seconds"), round(lat, 6), round(lng, 6), animal, incident,
            rules.urgency_for(animal, incident), rules.agency_for(animal, incident),
            status, None, "simulated", None, None, resolved_at, 1,
        ))
    if reset:
        query("DELETE FROM reports WHERE simulated=1", fetch="none")
    query_many(
        """INSERT INTO reports (created_at, lat, lng, animal, incident, urgency, agency, status,
           confidence, ai_mode, photo, note, resolved_at, simulated)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", rows)
    print(f"Inserted {len(rows)} simulated reports.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=450)
    p.add_argument("--reset", action="store_true", help="delete existing simulated rows first")
    p.add_argument("--if-empty", action="store_true", help="skip if simulated data exists")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    main(a.n, a.reset, a.if_empty, a.seed)
