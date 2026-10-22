# WildWatch Elmina

## 🔗 Try it live
- **Resident app:** https://wildwatch-elmina.onrender.com/app (open on your phone; you can add it to your home screen)
- **Management dashboard:** https://wildwatch-elmina.onrender.com/dashboard (login key is "elmina2026")

Prototype for the Sime Darby Property University Hackathon 2026 (Township of Tomorrow, Planet track).

Residents report animal sightings (monkeys, stray dogs, snakes, wild boar, monitor lizards) from their phone. AI identifies the animal, the system routes the report to the right party (PERHILITAN, MBSA or township management) and gives safety tips. Township management sees every report on a hotspot dashboard and gets suggested preventive actions.

**Stack:** FastAPI (Python) · SQLite (built-in, no setup) · MobileNetV2 image classifier (TFLite) · Leaflet + Chart.js · installable PWA · hosted on Render.

## What's inside

| Part | File | What it does |
|---|---|---|
| Resident app (PWA) | `frontend/report.html` | Photo, AI result, incident type, GPS or map pin, report ID, safety tips, status tracking, install to home screen |
| Management dashboard | `frontend/dashboard.html` | Login, heatmap with filters, KPI cards, top 5 hotspots with suggested actions, charts, case tracker, auto-refresh every 10 s |
| Backend API | `backend/app.py` | Classify, create, track, update and list reports, stats, hotspot detection, health check |
| Database | `backend/db.py` | Saves reports in one SQLite file, `data/wildwatch.db`; photos go in `data/uploads/` |
| Google Sheet copy | `backend/sheets.py`, `google_sheets/Code.gs` | Optional permanent copy of live reports, reloaded after a restart |
| Routing rules | `backend/rules.py` | Urgency, agency, safety tips and suggested actions |
| AI classifier | `backend/classifier.py` | Runs the trained model; demo mode (resident picks the animal) until a model exists |
| Dataset helper | `training/prepare_dataset.py` | Merges downloaded photos into the 6 class folders, cleans, de-duplicates, balances and zips them |
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

## 2b. Google Sheet copy (keeps live reports through restarts)

Optional, about 10 minutes. Every live report is copied to a Google Sheet, status changes are copied too, and when Render restarts the app loads the reports back from the sheet, with the same report IDs. Photos are not kept (they stay on Render's temporary disk). Sample reports are not copied.

1. **Create the sheet:** go to sheets.new, and name it, for example, `WildWatch reports`.
2. **Add the script:** in the sheet, click **Extensions > Apps Script**. Delete what's in the editor, paste the whole of `google_sheets/Code.gs`, and change `change-this-secret` on the `SECRET` line to your own long random phrase. Click the **Save** icon.
3. **Publish it:** click **Deploy > New deployment**. Click the gear next to "Select type" and choose **Web app**. Set **Execute as: Me** and **Who has access: Anyone**, then **Deploy**.
4. **Allow access:** Google asks you to authorise it. Choose your account; if you see "Google hasn't verified this app", click **Advanced > Go to (project name) (unsafe)**, then **Allow**. This is your own script, so it's safe.
5. **Copy the Web app URL** it shows (it ends in `/exec`).
6. **Add it to Render:** on the service page, open **Environment** and add two variables: `SHEETS_URL` = the Web app URL, and `SHEETS_SECRET` = the same phrase you typed in step 2. Save, and Render redeploys.
7. **Check:** open `/api/health`; it should show `"google_sheet": true`. Send a test report: within a few seconds a row appears in the sheet's **Reports** tab.

Keep the Web app URL and the secret private, like a password: only put them in Render's settings, never in GitHub or the slides. If you edit `Code.gs` later, use **Deploy > Manage deployments > Edit > New version** so the same URL keeps working.

To get an Excel file from the sheet: **File > Download > Microsoft Excel (.xlsx)**. Or use the dashboard's **Export to Excel** button.

**Export to Excel:** the dashboard's **Export to Excel** button (top right, after signing in) downloads every report on the server, sample and live, as an `.xlsx` file with filters on each column.

## 3. Train the AI model

### a. Download the photos

| Dataset (Kaggle) | What we use | Folder after unzipping |
|---|---|---|
| **Wildlife_detetion_yolo_74_class** (YOLO format) | monkey, dog, wild_boar, the 3 rattlesnake classes (→ snake), and cat, squirrel, crow, chicken, sparrow, mongoose, rabbit, duck (→ other). "Wild Dog" is left out on purpose. | `downloads/wildlife74` (the folder with `data.yaml`) |
| A **cats, dogs and snakes** image classification dataset (about 1,000 photos each) | only the dogs and snakes, for more dogs and more varied snakes | `downloads/catdogsnake` |
| **iNaturalist**: Asian water monitor (*Varanus salvator*), Malaysia | monitor lizard (biawak) | export CSV saved as `downloads/inat_biawak.csv` |
| **iNaturalist**: Common house gecko (*Hemidactylus frenatus*), Malaysia | cicak, added to `other` so the AI learns a house lizard isn't a case | export CSV saved as `downloads/inat_cicak.csv` |

**iNaturalist export:** sign in at inaturalist.org, open the observation export page (Explore, then Download), and filter by species name, place **Malaysia**, **Research Grade**, **has photos**, and photo licences CC0, CC-BY, CC-BY-NC if offered. Keep the **image_url** column ticked, export, and save the CSV with the name above. The script downloads the photos the first time (a few minutes) and reuses them on later runs.

Put both under a `downloads/` folder in the project. `downloads/` is in `.gitignore`, so the photos are never uploaded to GitHub. Check each dataset's licence on its Kaggle page.

### b. Build the dataset with the helper script

```bat
pip install pillow
python training/prepare_dataset.py ^
  --yolo downloads/wildlife74 ^
  --add dog downloads/catdogsnake/dogs ^
  --add snake downloads/catdogsnake/snakes ^
  --inat monitor_lizard downloads/inat_biawak.csv ^
  --inat other downloads/inat_cicak.csv ^
  --max 250 --zip
```
(On Windows Command Prompt, `^` continues a line; on Mac or Colab use `\`. Change the dogs/snakes paths if the folders are named differently after unzipping.)

For the YOLO dataset the script reads `data.yaml`, cuts out every boxed animal (with a little space around it), and sorts it into our 6 classes using `YOLO_MAP` at the top of the script. Then, for every source, it converts to JPG, skips broken and tiny crops, removes duplicates (including the same photo in two classes), shrinks large photos, and keeps at most 250 per class, **taking an equal share from each source** (so `other` gets a mix of all 8 animals plus cicak). It creates `dataset.zip`, and the summary shows how many photos each class got and where they came from. Then **open the folders and delete any obviously wrong photo** (a blank crop, a person, the wrong animal) and run `python training/prepare_dataset.py --zip-only` to remake the zip. Don't run the full command again: it rebuilds the folders and the deleted photos come back.

Need more photos of one animal later (for example local macaques from Roboflow or iNaturalist)? Add another `--add CLASS FOLDER` or `--inat CLASS CSV` line; see `python training/prepare_dataset.py --help`.

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
