"""Urgency, routing and safety-tip rules for each report."""

ANIMALS = ["monkey", "dog", "snake", "wild_boar", "monitor_lizard", "other"]
INCIDENTS = ["sighting", "bins", "entered_home", "damage", "attack"]

ANIMAL_LABELS = {
    "monkey": "Monkey (macaque)",
    "dog": "Stray dog",
    "snake": "Snake",
    "wild_boar": "Wild boar",
    "monitor_lizard": "Monitor lizard (biawak)",
    "other": "Other animal",
}

INCIDENT_LABELS = {
    "sighting": "Sighting only",
    "bins": "Rummaging bins",
    "entered_home": "Entered home compound",
    "damage": "Property damage",
    "attack": "Attack or bite",
}

AGENCIES = {
    "perhilitan": "PERHILITAN (Wildlife Department)",
    "council": "Majlis Bandaraya Shah Alam (MBSA)",
    "township": "Elmina Township Management",
    "emergency": "Bomba / emergency (999)",
}

TIPS = {
    "monkey": [
        "Do not feed or make eye contact with the monkey.",
        "Close windows and doors, and keep food out of sight.",
        "Keep bin lids shut and secured.",
    ],
    "dog": [
        "Stay calm, do not run or turn your back.",
        "Keep children indoors until the dog leaves.",
        "If bitten, wash the wound with soap for 15 minutes and see a doctor.",
    ],
    "snake": [
        "Keep at least 2 metres away and do not try to catch it.",
        "Keep children and pets away from the area.",
        "Watch where it goes so responders can find it.",
    ],
    "wild_boar": [
        "Move away slowly and give it space to leave.",
        "Do not corner or chase it.",
        "Keep food waste sealed overnight.",
    ],
    "monitor_lizard": [
        "Keep at least 2 metres away. Do not corner it, it can whip its tail and bite.",
        "Keep children, pets and chickens away, and close doors so it can leave on its own.",
        "Do not try to catch it. Watch where it goes so responders can find it.",
    ],
    "other": [
        "Keep a safe distance and do not touch the animal.",
        "Keep children and pets indoors.",
    ],
}


def urgency_for(animal: str, incident: str) -> str:
    if incident == "attack":
        return "high"
    if animal == "snake" and incident in ("entered_home", "damage"):
        return "high"
    if incident in ("entered_home", "damage") or animal in ("snake", "wild_boar", "monitor_lizard"):
        return "medium"
    return "low"


def agency_for(animal: str, incident: str) -> str:
    if animal in ("snake", "monitor_lizard") and incident in ("entered_home", "attack"):
        # Bomba removes snakes and monitor lizards from homes.
        return "emergency"
    if incident == "bins" and animal in ("monkey", "dog", "wild_boar", "monitor_lizard", "other"):
        # The cause is waste management, which township management can fix.
        return "township"
    if animal in ("monkey", "snake", "wild_boar", "monitor_lizard"):
        return "perhilitan"
    if animal == "dog":
        return "council"
    return "township"


def suggestion_for(animal: str, incident: str, period: str) -> str:
    if animal == "monkey" and incident == "bins":
        return f"Install monkey-proof bins and move collection before the {period} peak."
    if animal == "monkey":
        return "Put up 'Do not feed' signs and trim trees touching rooftops."
    if animal == "dog":
        return "Request an MBSA neutering and capture programme, and check for feeding spots."
    if animal == "wild_boar":
        return "Seal food waste overnight and inspect the fence line at the forest edge."
    if animal == "snake":
        return "Clear overgrown grass and drains, and brief residents on snake safety."
    if animal == "monitor_lizard":
        return "Cover drains and pond edges, and remind residents to secure chicken coops, pet food and waste."
    return "Inspect the area and brief residents."


def period_of(hour: int) -> str:
    if 5 <= hour < 10:
        return "early morning"
    if 10 <= hour < 16:
        return "daytime"
    if 16 <= hour < 20:
        return "evening"
    return "night"
