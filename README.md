# 🐾 WildWatch Elmina

**Smart stray animal detection for a safer, cleaner Elmina.**
Sime Darby Property University Hackathon 2026 · Township of Tomorrow · Planet track

## 🔗 Try it live

| | Link |
|---|---|
| 📱 **Resident app** | https://wildwatch-elmina.onrender.com/app (best on a phone; you can add it to your home screen) |
| 🖥️ **Management dashboard** | https://wildwatch-elmina.onrender.com/dashboard (login key is in our submission form) |
| 🎬 **Demo video** | *add your video link here* |

---

## The problem

City of Elmina is a green township built right beside the forest, so wildlife is part of daily life.

- **Safety risk:** aggressive strays and wild animals threaten residents, especially children.
- **Dirty streets:** monkeys raid bins and leave rubbish everywhere.
- **Slow, scattered reporting:** residents don't know who to call, reports come in with no photo, no location and no tracking, and help arrives late.

## Our solution

**Snap it. Pin it. Track it.**

1. **Snap it:** a resident takes a photo. Our AI identifies the animal in seconds and shows what to do right now.
2. **Pin it:** GPS adds the exact spot. The resident picks what's happening (for example "rummaging bins") and sends it, with no sign-up. They get a report number instantly.
3. **Track it:** the system decides the urgency and the right agency. Township management sees the case on a live dashboard, forwards it and updates the status, and the resident can follow it to "Resolved".

The dashboard also finds **hotspots**, places where the same problem keeps happening, and suggests fixes such as monkey-proof bins, so the township can prevent problems instead of only reacting.

## Key features

**Resident app** (installable web app, works on any phone)
- Photo report with an AI animal result and confidence score
- Automatic GPS location, or drop a pin on the map
- Instant safety tips for each animal, and a one-tap 999 button
- "My reports" to track each case's status
- Map of recent sightings nearby (locations rounded to about 100 m for privacy)
- English and Bahasa Melayu

**Management dashboard**
- Live alert when a new report comes in
- Heatmap and top 5 hotspots with suggested preventive actions
- KPIs with trends, plus charts by animal, situation and hour of day
- Response time for each agency against its target
- Open cases sorted by urgency: update the status, or forward a case by WhatsApp or email with a Google Maps link
- Export to Excel and print

## How the AI works

- **Model:** MobileNetV2 (transfer learning), exported to TensorFlow Lite so it runs on a small free server.
- **Classes:** monkey, stray dog, snake, wild boar, monitor lizard, and **other**. "Other" covers non-animal photos and other animals, so a random photo isn't forced into the wrong animal.
- **Training data:** public datasets from Kaggle, plus research-grade **iNaturalist photos from Malaysia** for local species such as macaques, wild boar and the Asian water monitor (biawak).
- **Results:** *XX% accuracy on test photos (add your latest number and confusion matrix).*
- **Safety net:** when the AI isn't confident enough, the resident is asked to pick the animal, so a report is never blocked.

## Who handles what

| Situation | Sent to |
|---|---|
| Any animal rummaging bins | Elmina township management (a waste problem the township can fix) |
| Snake or monitor lizard inside a home, or any attack | Fire and Rescue (Bomba) |
| Monkey, snake, wild boar or monitor lizard (other cases) | PERHILITAN (Wildlife Department) |
| Stray dog | MBSA (Shah Alam City Council) |

## Tech stack

FastAPI (Python) · SQLite · TensorFlow Lite (MobileNetV2) · Leaflet maps · installable PWA · Google Sheets backup · hosted on Render

## Project structure

| Path | What it is |
|---|---|
| `frontend/app.html` | Resident app |
| `frontend/admin.html` | Management dashboard |
| `backend/app.py` | API: classify photos, create, track and update reports, stats, hotspots |
| `backend/rules.py` | Urgency, agency routing, safety tips and suggested actions |
| `backend/classifier.py` | Runs the trained AI model |
| `backend/generate_data.py` | Simulated sample reports around Elmina for the dashboard |
| `model/` | Trained model (`animal_model.tflite`) and class labels |
| `training/` | Dataset builder, training script and Google Colab notebook |
| `google_sheets/Code.gs` | Optional Google Sheet copy of live reports |

## Data note

The dashboard includes **simulated sample reports** (marked "SAMPLE") so the hotspot and chart features can be shown. Live reports from the app appear alongside them. Passwords and keys are kept in the server settings, never in this repository.

---

<details>
<summary><b>Run it on your own computer</b></summary>

Needs Python 3.

```bash
pip install -r requirements.txt
python -m backend.generate_data --reset
uvicorn backend.app:app --reload
```

- Resident app: http://localhost:8000/app
- Dashboard: http://localhost:8000/dashboard (local key: `elmina2026`)
- Health check: http://localhost:8000/api/health

On Windows the AI runtime isn't available, so the app runs in demo mode locally (the resident picks the animal) unless you `pip install tensorflow`. The live link runs the full AI.

</details>

<details>
<summary><b>Deploy and keep live reports</b></summary>

- **Render:** the repo includes `render.yaml`. Create a Blueprint from the repo on render.com and set `ADMIN_KEY` (the dashboard password) in the service's Environment settings.
- **Google Sheet backup (optional):** paste `google_sheets/Code.gs` into a Google Sheet's Apps Script, deploy it as a web app, and set `SHEETS_URL` and `SHEETS_SECRET` in Render's Environment settings. Live reports are then copied to the sheet and reloaded after a restart.

</details>

<details>
<summary><b>Retrain the AI</b></summary>

1. Build the dataset: `python training/prepare_dataset.py --help` shows how to combine Kaggle folders and iNaturalist exports into the 6 classes.
2. Train on Google Colab with `training/WildWatch_training.ipynb` and `training/train.py`.
3. Copy `animal_model.tflite` and `labels.json` into `model/`.

</details>

---

## Team

- **Anis Dayana** (Team Leader)
- **Nurul Ain Najihah**
- **Nur Aaliyah Nabihah**
