# WildWatch Elmina

Prototype for the Sime Darby Property University Hackathon 2026 (Township of Tomorrow, Planet track).

Residents report animal sightings (monkeys, stray dogs, snakes, wild boar) from their phone. AI identifies the animal, the system routes the report to the right party (PERHILITAN, MBSA or township management) and gives safety tips. Township management sees every report on a hotspot dashboard and gets suggested preventive actions.

**Stack:** FastAPI (Python) · SQLite (built-in, no setup) · MobileNetV2 image classifier (TFLite) · Leaflet + Chart.js · installable PWA · hosted on Render.

## What's inside

| Part | File | What it does |
|---|---|---|
| Resident app (PWA) | `frontend/report.html` | Photo, AI result, incident type, GPS or map pin, report ID, safety tips, status tracking, install to home screen |
| Management dashboard | `frontend/dashboard.html` | Login, heatmap with filters, KPI cards, top 5 hotspots with suggested actions, charts, case tracker, auto-refresh every 10 s |
| Backend API | `backend/app.py` | Classify, create, track, update and list reports, stats, hotspot detection, health check |
| Database | `backend/db.py` | Saves reports in one SQLite file, `data/wildwatch.db`; photos go in `data/uploads/` |
| Routing rules | `backend/rules.py` | Urgency, agency, safety tips and suggested actions |
| AI classifier | `backend/classifier.py` | Runs the trained model; demo mode (resident picks the animal) until a model exists |
| Dataset helper | `training/prepare_dataset.py` | Merges downloaded photos into the 5 class folders, cleans, de-duplicates, balances and zips them |
| Training notebook | `training/WildWatch_training.ipynb` | Step-by-step Google Colab notebook: upload, train, check results, test a photo, download |
| Model training | `training/train.py` | MobileNetV2 transfer learning; saves the model, a TFLite copy, accuracy report and confusion matrix |
| Simulated data | `backend/generate_data.py` | 450 simulated reports around City of Elmina with built-in hotspot patterns |
| Deployment | `render.yaml` | Tells Render how to run the app online |

## 1. Run it locally

No accounts or database setup needed: reports are saved in a small file, `data/wildwatch.db`.

```bash
pip install -r requirements.txt
python -m backend.generate_data --reset
uvicorn backend.app:app --reload
```

- Resident app: http://localhost:8000/report
- Dashboard: http://localhost:8000/dashboard (admin key: `elmina2026`)
- Health check: http://localhost:8000/api/health shows whether the AI model is loaded

**Quicker ways to start it (after the one-time install above):**
- **VS Code:** open the `wildwatch` folder, press **F5**, and pick "Start WildWatch (dashboard)" or "(resident app)". The server starts and the page opens in your browser. Press **Shift + F5** to stop. Needs the Python extension (VS Code offers to install it).
- **Double-click `start.bat`:** a window opens running the server, and the dashboard opens in your browser. Close the window to stop.

Don't use "Open with Live Server" or "PHP Server: Open file in browser" for this project: those can't run the Python backend, so the pages won't work.

On Windows the small AI runtime isn't available, so the app runs in demo mode locally unless you `pip install tensorflow`. That's fine: the deployed version on Render runs the model.

## 2. Deploy on Render

1. Push this folder to a GitHub repository.
2. On render.com, sign in with GitHub, then **New + > Blueprint** and choose the repo. Render reads `render.yaml` and sets up the service.
3. When asked, type `ADMIN_KEY`: the dashboard password you choose.
4. Wait for the deploy to finish, then open `https://<your-service>.onrender.com/api/health`. It should show `"ok": true`.

Your submission links:
- Resident app: `https://<your-service>.onrender.com/report`
- Dashboard: `https://<your-service>.onrender.com/dashboard`

Every push to GitHub redeploys automatically. On the free plan the server sleeps after about 15 minutes with no visitors and takes around a minute to wake, so open the link yourself shortly before judging.

**About saved data:** on Render's free plan, the server's files reset whenever it restarts (after sleeping, and on every update). Reports and photos that people submit live are cleared then, while the 450 sample reports are recreated automatically, so the dashboard is never empty. That's fine for the demo. A pilot would add a cloud database such as PostgreSQL so records are kept permanently.

## 3. Train the AI model

### a. Download the photos

| Class | Source | Folder to use after unzipping |
|---|---|---|
| `other` | Kaggle: Animal Image Dataset (90 different animals) | the `cat`, `squirrel`, `lizard`, `rat`, `crow`, `pigeon` folders |
| `wild_boar` | same 90-animals dataset, plus iNaturalist (below) | the `boar` folder |
| `dog` | Kaggle: Animals-10 | `raw-img/cane` (folder names are in Italian; *cane* = dog) |
| `snake` | Kaggle: a snake image dataset | the folder with the snake photos |
| `monkey` | iNaturalist: Long-tailed Macaque (below) | export CSV |

Check each dataset's licence on its Kaggle page, and look inside each folder after unzipping, since paths can differ slightly.

**iNaturalist (for monkey and wild boar):**
1. Sign in at inaturalist.org, open **Explore**, and search the species: *Long-tailed Macaque* (and separately *Wild Boar*).
2. Filter to **Research Grade** observations with photos. If the filter offers photo licences, choose CC0, CC-BY and CC-BY-NC, which allow reuse.
3. Use **Download** (the observations export page), keep the **image_url** column ticked, and export the CSV. Save it as `inat_macaque.csv` (or `inat_boar.csv`).

### b. Build the dataset with the helper script

```bash
pip install pillow
python training/prepare_dataset.py ^
  --add other downloads/animals90/cat --add other downloads/animals90/squirrel ^
  --add other downloads/animals90/lizard --add other downloads/animals90/rat ^
  --add wild_boar downloads/animals90/boar ^
  --add dog downloads/animals10/raw-img/cane ^
  --add snake downloads/snakes ^
  --inat monkey downloads/inat_macaque.csv --inat wild_boar downloads/inat_boar.csv ^
  --max 300 --zip
```
(On Windows Command Prompt, `^` continues a line; on Mac or Colab use `\`. Replace the paths with where you unzipped each download.)

The script copies photos into `dataset/monkey`, `dog`, `snake`, `wild_boar` and `other`, converts formats to JPG, skips broken and tiny images, removes duplicates (including resized copies and the same photo in two classes), shrinks large photos, keeps at most 300 per class, downloads the iNaturalist photos, and creates `dataset.zip`. The summary at the end shows how many photos each class got. Then **open the folders and delete any obviously wrong photo** (a cartoon, a person, the wrong animal).

### c. Train on Google Colab

1. Go to colab.research.google.com, **File > Upload notebook**, and choose `training/WildWatch_training.ipynb`.
2. Follow the notebook: turn on the GPU, upload `dataset.zip` and `training/train.py`, then press play on each step. Training takes about 15 to 30 minutes.
3. The last step downloads `wildwatch_model.zip`.

### d. Add the model to the app

1. Copy `animal_model.tflite` and `labels.json` from `wildwatch_model.zip` into the project's `model/` folder.
2. Upload them to GitHub. Render redeploys and `/api/health` shows `"ai_model": true`.
3. Put `confusion_matrix.png` and the accuracy from `report.txt` in the pitch deck.

If confidence is below 60% (`CONFIDENCE_THRESHOLD`), the resident is asked to pick the animal.

## Install as an app (PWA)

- **Android (Chrome):** an "Install" banner appears at the top of the resident app. Tap it.
- **iPhone (Safari):** the banner shows the steps: tap Share, then "Add to Home Screen".

Installing needs HTTPS, so it works on the Render link and on `localhost`, but not on a plain local IP address.

## Demo script (for the 5-minute video)

1. Open the dashboard on a laptop next to a phone.
2. On the phone, install the app from the link, then open it from the home screen.
3. Send a monkey photo, pick "Rummaging bins", use the location.
4. Show the result: animal, agency, urgency, safety tips, report ID.
5. Within 10 seconds the dashboard shows "New report received" and a new dot on the map.
6. Click hotspot 1 to show the pattern and the suggested action.
7. Change the report status to resolved, then check it on the phone with "Track report".

## Data note

Dashboard data is simulated for the prototype and flagged `simulated=1` (shown as "sim"). On Render it is created each time the app starts. Locally, run `python -m backend.generate_data --reset` to regenerate it, for example after changing the hotspot locations. Hotspot coordinates in `generate_data.py` are approximate locations around City of Elmina; adjust them to real bins, parks and forest edges before the pitch.
