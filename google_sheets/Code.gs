/**
 * WildWatch Elmina: Google Sheet storage for live reports.
 *
 * Paste this whole file into the sheet's Apps Script editor (Extensions > Apps Script),
 * change SECRET below, then Deploy > New deployment > Web app
 * (Execute as: Me, Who has access: Anyone). Full steps: README, "Google Sheet copy".
 *
 * The app adds a row for each new report, updates the row when its status changes,
 * and reads all rows back after the server restarts.
 */

// Change this to your own long random phrase, and put the same value in Render as SHEETS_SECRET.
const SECRET = 'change-this-secret';

const SHEET_NAME = 'Reports';
const HEADERS = ['id', 'created_at', 'lat', 'lng', 'animal', 'incident', 'urgency', 'agency',
                 'status', 'confidence', 'ai_mode', 'photo', 'note', 'resolved_at'];

function getSheet_() {
  const ss = SpreadsheetApp.getActiveSpreadsheet();
  let sh = ss.getSheetByName(SHEET_NAME);
  if (!sh) sh = ss.insertSheet(SHEET_NAME);
  if (sh.getLastRow() === 0) {
    sh.getRange(1, 1, 1, HEADERS.length).setValues([HEADERS]).setFontWeight('bold');
    sh.setFrozenRows(1);
    // Keep dates and text exactly as the app sends them (no automatic date conversion).
    sh.getRange(1, 1, sh.getMaxRows(), HEADERS.length).setNumberFormat('@');
  }
  return sh;
}

function reply_(obj) {
  return ContentService.createTextOutput(JSON.stringify(obj)).setMimeType(ContentService.MimeType.JSON);
}

function findRow_(sh, id) {
  const last = sh.getLastRow();
  if (last < 2) return 0;
  const ids = sh.getRange(2, 1, last - 1, 1).getValues();
  for (let i = 0; i < ids.length; i++) {
    if (String(ids[i][0]) === String(id)) return i + 2;
  }
  return 0;
}

// The app reads all reports back (after a restart).
function doGet(e) {
  if (!e || !e.parameter || e.parameter.secret !== SECRET) return reply_({ ok: false, error: 'unauthorized' });
  const sh = getSheet_();
  const last = sh.getLastRow();
  if (last < 2) return reply_({ ok: true, rows: [] });
  const values = sh.getRange(2, 1, last - 1, HEADERS.length).getValues();
  const rows = values
    .filter(function (r) { return r[0] !== ''; })
    .map(function (r) {
      const o = {};
      HEADERS.forEach(function (h, i) {
        const v = r[i];
        o[h] = v === '' ? null : (v instanceof Date ? v.toISOString() : v);
      });
      return o;
    });
  return reply_({ ok: true, rows: rows });
}

// The app adds a report, or updates a report's status.
function doPost(e) {
  let body;
  try {
    body = JSON.parse(e.postData.contents);
  } catch (err) {
    return reply_({ ok: false, error: 'bad request' });
  }
  if (body.secret !== SECRET) return reply_({ ok: false, error: 'unauthorized' });

  const lock = LockService.getScriptLock();
  lock.waitLock(20000);
  try {
    const sh = getSheet_();
    if (body.action === 'add') {
      const r = body.report || {};
      if (r.id === undefined || r.id === null) return reply_({ ok: false, error: 'missing id' });
      if (findRow_(sh, r.id)) return reply_({ ok: true, duplicate: true });
      const row = HEADERS.map(function (h) {
        const v = r[h];
        return v === undefined || v === null ? '' : String(v);
      });
      const target = sh.getLastRow() + 1;
      sh.getRange(target, 1, 1, HEADERS.length).setNumberFormat('@').setValues([row]);
      return reply_({ ok: true });
    }
    if (body.action === 'status') {
      const rowNum = findRow_(sh, body.id);
      if (!rowNum) return reply_({ ok: false, error: 'report not found' });
      sh.getRange(rowNum, HEADERS.indexOf('status') + 1).setValue(String(body.status || ''));
      if (body.resolved_at) {
        sh.getRange(rowNum, HEADERS.indexOf('resolved_at') + 1).setNumberFormat('@').setValue(String(body.resolved_at));
      }
      return reply_({ ok: true });
    }
    return reply_({ ok: false, error: 'unknown action' });
  } finally {
    lock.releaseLock();
  }
}
