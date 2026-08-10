# DeepShield AI — Current State

**Baseline snapshot.** Everything below was read from the running system on
the date shown, not from memory. Figures marked *measured* come from a run
recorded in this document.

| | |
|---|---|
| Snapshot date | 2026-08-10 (updated after Phase 5) |
| Commit | `ab0103d983ed5b272f964ee3cc750a91116b3c8c` (`ab0103d`) |
| Commit subject | Run the model as ONNX: 1.9 GB backend becomes 197 MB |
| Branch | `main`, clean working tree, 31 commits |
| Remote | https://github.com/kumudranjan6127-debug/deepshield-ai |
| Machine used for measurements | Windows 10, 4 CPU cores, CPU-only |

---

## 1. Model in production

**MobileNetV3-Large · ONNX · 224×224 · classes `["fake", "real"]` · V3-Max.**
This is the single source of truth; see `MODEL_CARD.md` for the full card.

| Artifact | Size | Role |
|---|---|---|
| `models/deepshield.onnx` | 16.8 MB | **The live model.** Loaded by default |
| `models/deepshield.onnx.json` | ~700 B | Identity block, classes, input size, normalisation, metrics |
| `models/deepshield_mobilenetv3.pth` | 17.0 MB | Same weights, PyTorch. Fallback only |
| `models/face_detection_yunet.onnx` | 232 KB | YuNet face detector |
| `models/archive/v1_baseline.pth` | 6.2 MB | History |
| `models/archive/v2_heavy.pth` | 6.2 MB | History |
| `models/archive/v3_max.pth` | 17.0 MB | History — same weights as the live model |

`models/deepshield.onnx.json` (verbatim):

```json
{
  "model_name": "DeepShield",
  "architecture": "MobileNetV3-Large",
  "version": "V3-Max",
  "runtime": "ONNX",
  "arch": "mobilenet_v3_large",
  "classes": ["fake", "real"],
  "input_size": 224,
  "normalize": { "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225] },
  "val_accuracy": 99.9,
  "robust_val_accuracy": 99.18,
  "tpdn_accuracy": 100.0,
  "dfdc_accuracy": null,
  "test_accuracy": null,
  "trained_on": "V3-Max multi-generator: SG1 + TPDN/SG2 + diffusion, 10 epochs, large",
  "source_checkpoint": "deepshield_mobilenetv3.pth"
}
```

### Backend selection

`backend/inference.py` picks a runtime at load:

1. **`onnx`** — chosen when `deepshield.onnx` **and** its `.json` exist. Runs
   through `cv2.dnn`. No PyTorch needed. *This is the active path.*
2. **`torch`** — used only when no ONNX export is present and torch is
   importable.
3. Neither available → `engine_available()` is false, the app serves the
   simulated engine and labels itself "Simulated (demo)".

Both backends share preprocessing and were verified to agree to **3.24e-07**
on output probabilities (`scripts/export_onnx.py` re-checks this on export).

---

## 2. Inference pipeline (what one image goes through)

```
upload → staged in uploads/ by POST /api/upload
   ↓
image capped at 1024px longest side (INTER_AREA)
   ↓
YuNet face detection → largest face, 0.35 margin crop
   ↓  (no face found → whole frame is used)
JPEG q88 round-trip                  ← compression-domain normalisation
   ↓
resize 224×224 (PIL BILINEAR) → /255 → ImageNet normalise → CHW
   ↓
forward ×2 (image + horizontal mirror), probabilities averaged   ← TTA
   ↓
optional verifiers (off by default)
   ↓
verdict + confidence + risk + occlusion heatmap
   ↓
uploaded file deleted
```

**Video:** one frame per second (`frameRate` setting, max 60 frames), each
frame through the same path, probabilities averaged. Verifiers and heatmap are
not run per-frame.

**Combining votes** (`inference.analyze_file`): own model weight **0.75**, the
remaining 0.25 split across verifiers. Verifiers may raise the score only when
**all** of them are ≥ 0.85. `disputed` is set when any vote disagrees with the
final verdict.

---

## 3. HTTP API

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | Serves `frontend/landing.html` |
| GET | `/<path>` | Static files from `frontend/` only |
| GET | `/api/health` | Engine, backend, model identity, metrics |
| POST | `/api/upload` | Stages a file, returns `uploadId` |
| POST | `/api/analyze` | Runs the analysis |
| POST | `/api/feedback` | Records a thumbs up/down (no media) |

### GET /api/health — live response

```json
{
  "ok": true,
  "status": "ok",
  "engine": "live",
  "model_name": "DeepShield",
  "architecture": "MobileNetV3-Large",
  "version": "V3-Max",
  "runtime": "ONNX",
  "input_size": 224,
  "classes": ["fake", "real"],
  "backend": "onnx",
  "checkpoint": "deepshield.onnx",
  "arch": "mobilenet_v3_large",
  "params": "5.4M",
  "val_accuracy": 99.9,
  "tpdn_accuracy": 100,
  "test_accuracy": null,
  "dfdc_accuracy": null,
  "trained_on": "V3-Max multi-generator: SG1 + TPDN/SG2 + diffusion, 10 epochs, large",
  "verifiers": false,
  "calibrated": false,
  "certainty_bands": [
    { "from": 90, "to": 100, "key": "very_strong",  "label": "Very strong evidence" },
    { "from": 70, "to":  90, "key": "strong",       "label": "Strong evidence" },
    { "from": 30, "to":  70, "key": "uncertain",    "label": "Uncertain" },
    { "from":  0, "to":  30, "key": "low_evidence", "label": "Low evidence" }
  ],
  "model": {
    "model_name": "DeepShield", "architecture": "MobileNetV3-Large",
    "version": "V3-Max", "runtime": "ONNX", "input_size": 224,
    "name": "MobileNetV3-Large", "params": "5.4M",
    "input": "224 × 224", "backend": "ONNX", "device": "CPU"
  }
}
```

Every field is read from `models/deepshield.onnx.json` at load; no model
fact is written twice. `scripts/model_test.py` asserts the metadata file,
the loaded engine and this response agree.

### POST /api/analyze

Accepts either multipart (`file`, `fileType`, `frameRate`) or JSON
(`uploadId` | `url`, `fileName`, `fileType`, `fileSize`, `frameRate`).

Live response for an image (heatmap abbreviated):

```json
{
  "ok": true,
  "prediction": "real",
  "confidence": 94,
  "riskLevel": "Low",
  "risk": "low",
  "certainty": "very_strong",
  "framesAnalyzed": 1,
  "processingTime": 831,
  "model": "MobileNetV3-Large",
  "device": "CPU",
  "completedAt": "2026-08-10T14:14:03.589570+00:00",
  "disputed": false,
  "ensemble": [ { "model": "MobileNetV3 (ours)", "pFake": 0.0612, "weight": 1 } ],
  "explain": {
    "focusRegion": "the eye region",
    "method": "occlusion sensitivity",
    "note": "Prediction was most sensitive to the eye region.",
    "heatmapDataUrl": "data:image/jpeg;base64,… (11,291 chars)"
  }
}
```

**Contract notes**
- `prediction` ∈ `"real" | "deepfake"`; `confidence` is an integer percent of
  the winning class; `riskLevel` ∈ `"Low" | "Medium" | "High"`.
- `ensemble[0]` is always our own model. `pFake` is its raw probability.
- `certainty` is the evidence band for `confidence`; `risk` is `riskLevel`
  lowercased. Both are additive — `riskLevel` is what the pages read, and
  the band table is published by `/api/health` so no threshold is written
  down in the browser. The model is uncalibrated, so `confidence` ranks
  evidence and does not estimate a frequency.
- `explain` is absent for video scans and may be `null` if the heatmap fails.
- `POST /api/feedback` requires a boolean `agree`; anything else returns 400.
- Failures share one shape: `{"ok": false, "error": "…", "error_code": "…"}`.
  Codes in use: `NO_FILE`, `BAD_TYPE`, `BAD_MIME`, `BAD_MAGIC`,
  `EMPTY_FILE`, `CORRUPT_MEDIA`, `IMAGE_TOO_LARGE`, `IMAGE_TOO_SMALL`,
  `VIDEO_TOO_LONG`, `TOO_LARGE`, `BLOCKED_URL`, `INSECURE_URL`,
  `URL_NOT_VIDEO`, `BAD_FIELD`, `RATE_LIMITED` (429), `BUSY` (503),
  `INVALID_INPUT`, `INTERNAL`. A missing **page** stays a normal HTML 404;
  only `/api/*` returns JSON errors.

---

## 4. Request handling and security

Every request passes the same gates, cheapest first:

```
rate limit (5/min per client)
   ↓
upload:  MAX_CONTENT_LENGTH → extension → MIME → magic bytes
   ↓                        → decode → dimensions / duration
url:     https → DNS → every resolved address must be public
   ↓            → each redirect re-validated
concurrency gate (2 analyses at once, then 503)
   ↓
inference
```

Blocked outbound destinations: loopback, private (10/8, 172.16/12,
192.168/16), link-local (169.254/16 — cloud metadata), reserved,
multicast, and the IPv6 equivalents including `::1`, `fe80::/10`,
`fc00::/7` and `::ffff:` mapped addresses.

Staged uploads are swept after 30 minutes by a background thread and at
startup; an analysed file is deleted as soon as the verdict is returned.

Implemented in `backend/security.py`; limits live in `config.py`.

## 5. Frontend

Framework-free HTML/CSS/JS. 11 pages:

`landing.html` (served at `/`) · `index.html` (splash) · `login.html` ·
`dashboard.html` · `upload-image.html` · `upload-video.html` ·
`processing.html` · `result.html` · `report.html` · `about.html` ·
`settings.html`

**Flow:** landing → splash → login (or guest) → dashboard → upload →
processing → result → report.

**Global `DS` namespace** (`assets/js/utils.js`, `api.js`, `components.js`):
`DS.util` · `DS.store` · `DS.session` · `DS.history` · `DS.settings` ·
`DS.auth` · `DS.api` · `DS.server` · `DS.toast` · `DS.modal` · `DS.sidebar` ·
`DS.shell` · `DS.icons` · `DS.glare`

**Browser storage**

| Key | Store | Contents |
|---|---|---|
| `ds_user` | local | `{name, email, loggedInAt}` |
| `ds_scan` | session | the scan in flight |
| `ds_history` | local | completed scans, newest first, capped at 50 |
| `ds_settings` | local | `{frameRate, threshold, reducedMotion, autoDelete, effects}` |
| `ds_feedback` | local | verdict ratings — **not registered in `DS.KEYS`**, see `KNOWN_ISSUES.md` #2 |

**Engine mode:** `DS.api.MODE = 'auto'` — probes `/api/health` once per page.
Live engine → real analysis; otherwise the simulated engine, labelled
"Simulated (demo)" on the dashboard and in Settings.

---

## 6. Dependencies

**Runtime (`requirements.txt`)** — the lean install, ~197 MB:

```
flask>=3.1
opencv-python>=4.8
pillow>=10.0
numpy>=1.24
```

**Present in this development venv** (a superset — training and export tools):

| Package | Version |
|---|---|
| Flask | 3.1.3 |
| opencv-python | 5.0.0.93 |
| numpy | 2.4.4 |
| pillow | 12.2.0 |
| torch | 2.13.0+cpu |
| torchvision | 0.28.0+cpu |
| transformers | 5.14.1 |
| onnx / onnxscript | 1.22.0 / 0.7.1 |
| Werkzeug / Jinja2 | 3.1.8 / 3.1.6 |

Python 3.14.7 · Node v24.17.0 (Node runs only `scripts/ds.js` and `serve.js`).

**Optional extras:** `torch torchvision onnx onnxscript` for training/export;
`transformers torch` for the verifiers; both unnecessary to run the app.

### Environment switches

| Variable | Default | Effect |
|---|---|---|
| `PORT` | 5000 | Backend port |
| `DS_DEBUG` | off | `1` enables Flask's reloader (spawns a second process) |
| `DS_ENGINE` | unset | `echo` forces the simulated engine — verified working |
| `DS_VERIFIERS` | off | `1` loads the two HuggingFace verifiers |

---

## 7. Measured performance

Measured on this machine (4 cores, CPU-only), ONNX backend, verifiers off:

| Operation | Time |
|---|---|
| Model load | 0.42 s |
| Image, first call | 0.94 s |
| Image, warm (avg of 5) | **0.61 s** (min 0.55, max 0.66) |
| Video, 8 sampled frames | 1.45 s |
| Video, 13 sampled frames | 1.62 s |
| Peak RSS after 4 analyses | **199 MB** |

Image timings include the occlusion heatmap (36 extra forward passes,
chunked 8 at a time).

**Footprint:** backend dependencies 197 MB, model 16.8 MB, peak RAM ~200 MB —
within the 512 MB free hosting tiers.

---

## 8. Accuracy on the held test images

Nine images kept outside training: five thispersondoesnotexist (StyleGAN2)
faces and one authentic press portrait at four resolutions.

| Image | Truth | Verdict | Own model P(fake) |
|---|---|---|---|
| random-person (1) | fake | fake 98% | 0.9757 |
| random-person (2) | fake | fake 97% | 0.9725 |
| random-person (3) | fake | fake 98% | 0.9766 |
| random-person (4) | fake | fake 97% | 0.9724 |
| random-person | fake | fake 97% | 0.9748 |
| press portrait 2687px | real | real 94% | 0.0612 |
| press portrait 1024px | real | real 98% | 0.0189 |
| press portrait 512px | real | real 98% | 0.0233 |
| press portrait 256px | real | real 98% | 0.0234 |

**9/9.** Identical under the PyTorch backend and in a clean environment with
no PyTorch installed. This is a small sample and is not a benchmark result —
see `MODEL_CARD.md` for what has and has not been evaluated.

## 9. Repository layout

```
frontend/          11 pages + assets (css, js, fonts, icons)
backend/           app.py (267 lines) · inference.py (565 lines)
models/            live ONNX + metadata + YuNet + archive/
training/          Kaggle/Colab notebooks, result graphs, test media
scripts/           ds.js (server control) · export_onnx.py (.pth → ONNX)
                   evaluate.py + ds_metrics.py (Phase 4 evaluation)
                   *_test.py (regression, security, model, metrics, split)
eval_data/         labelled images to evaluate on (gitignored)
docs/              this file, MODEL_CARD.md, KNOWN_ISSUES.md, DOCUMENTATION.md
uploads/ data/     runtime only, gitignored
venv/              local environment, gitignored
```

**Operating the server:** `npm start` · `npm run status` · `npm run stop` ·
`npm run restart`, or the `START-/STOP-DeepShield.bat` files.

---

## 10. Tests

```
python scripts/regression_test.py record|verify   behaviour has not changed
python scripts/security_test.py [--unit]          50 attacks, all refused
python scripts/model_test.py                      identity, reproducibility, parity
python scripts/metrics_test.py                    the metric arithmetic
python scripts/split_test.py                      the V4 split cannot leak
```

| Suite | Checks | Asserts | Needs server |
|---|---|---|---|
| regression | 23 | Engine outputs and every API response, field by field | yes |
| security | 50 | SSRF (v4/v6/DNS/redirects), oversized, forged extension, corrupt and empty media, malformed URLs, rate limit, concurrency, cleanup | yes |
| model | 40 | Identity agrees across file/engine/API; repeated runs identical; ONNX vs PyTorch within 1e-4 (measured 3.2e-08); certainty bands total, unambiguous and published identically | no |
| metrics | 63 | Hand-computed answers, plus ROC-AUC against brute-force pair counting, PR-AUC against a threshold walk, and calibration (Brier/ECE/MCE) against worked examples | no |
| split | 24 | Lifts the DFDC split out of the V4 notebook and runs it on a synthetic set whose leakage structure is known | no |

The regression baseline lives in `docs/regression_baseline.json`. Live
suites need the server running; `security_test.py` sends ~25 requests, so
start both it and the server with `DS_RATE_LIMIT=50` — and pass it to
`ds.js` itself, since a plain `restart` drops the variable and the suite
then throttles itself.

---

## 11. Evaluation

```
python scripts/evaluate.py                        scores eval_data/
python scripts/evaluate.py --seen sg1,sg2,dfdc    in-domain vs unseen
python scripts/evaluate.py --from-csv preds.csv   re-score a training run
python scripts/evaluate.py --conditions DIR       phone/screenshot/social variants
```

The report has four parts: the metric block, a per-source table, a threshold
sweep, and a **calibration** section — reliability diagram, ECE, MCE, Brier,
plus the observed accuracy and occupancy of every certainty band. That last
table is how a band label stops being a guess: a band called "Strong
evidence" that is right 61% of the time has the wrong cut point, and an empty
band cannot be produced at all.

`ds_metrics.py` holds the arithmetic — accuracy, precision, recall, F1,
specificity, ROC-AUC, PR-AUC, FPR, FNR, Brier, ECE, MCE — and nothing else computes a metric,
including the training notebook. The notebook emits raw scores as
`predictions_*.csv`; `--from-csv` turns them into the report. Kaggle numbers
and local numbers are therefore the same arithmetic, and any published figure
can be recomputed from the CSV behind it.

Scoring calls `inference.score_image`, the request path's own preprocessing
minus the heatmap. Undefined metrics print `n/a`: a ROC-AUC over one class is
not 0.5.

Drop data into `eval_data/` — see `eval_data/README.md` for the layout and
the identity-separation rules. The folder is gitignored apart from that file.
