# DeepShield AI — Current State

**Baseline snapshot.** Everything below was read from the running system on
the date shown, not from memory. Figures marked *measured* come from a run
recorded in this document.

| | |
|---|---|
| Snapshot date | 2026-08-10 |
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
| `models/deepshield.onnx.json` | 535 B | Classes, input size, normalisation, metrics |
| `models/deepshield_mobilenetv3.pth` | 17.0 MB | Same weights, PyTorch. Fallback only |
| `models/face_detection_yunet.onnx` | 232 KB | YuNet face detector |
| `models/archive/v1_baseline.pth` | 6.2 MB | History |
| `models/archive/v2_heavy.pth` | 6.2 MB | History |
| `models/archive/v3_max.pth` | 17.0 MB | History — same weights as the live model |

`models/deepshield.onnx.json` (verbatim):

```json
{
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
  "status": "ok",
  "engine": "live",
  "backend": "onnx",
  "checkpoint": "deepshield.onnx",
  "arch": "mobilenet_v3_large",
  "val_accuracy": 99.9,
  "tpdn_accuracy": 100,
  "test_accuracy": null,
  "dfdc_accuracy": null,
  "trained_on": "V3-Max multi-generator: SG1 + TPDN/SG2 + diffusion, 10 epochs, large",
  "verifiers": false,
  "model": {
    "name": "MobileNetV3-Small", "params": "2.5M", "backend": "PyTorch",
    "input": "224 × 224", "device": "CPU", "version": "1.0.0"
  }
}
```

> ⚠️ The nested `model` block is **stale** — it is a hardcoded constant in
> `backend/app.py` and still describes the V1/V2 era. `arch` and `checkpoint`
> are the accurate fields. See `KNOWN_ISSUES.md` #1.

### POST /api/analyze

Accepts either multipart (`file`, `fileType`, `frameRate`) or JSON
(`uploadId` | `url`, `fileName`, `fileType`, `fileSize`, `frameRate`).

Live response for an image (heatmap abbreviated):

```json
{
  "prediction": "real",
  "confidence": 94,
  "riskLevel": "Low",
  "framesAnalyzed": 1,
  "processingTime": 831,
  "model": "MobileNetV3-Small",
  "device": "CPU",
  "completedAt": "2026-08-10T14:14:03.589570+00:00",
  "disputed": false,
  "ensemble": [ { "model": "MobileNetV3 (ours)", "pFake": 0.0612, "weight": 1 } ],
  "explain": {
    "focusRegion": "the eye region",
    "method": "occlusion sensitivity",
    "note": "Model attention concentrated around the eye region.",
    "heatmapDataUrl": "data:image/jpeg;base64,… (11,291 chars)"
  }
}
```

> ⚠️ The top-level `"model"` field also comes from the stale constant.

**Contract notes**
- `prediction` ∈ `"real" | "deepfake"`; `confidence` is an integer percent of
  the winning class; `riskLevel` ∈ `"Low" | "Medium" | "High"`.
- `ensemble[0]` is always our own model. `pFake` is its raw probability.
- `explain` is absent for video scans and may be `null` if the heatmap fails.
- `POST /api/feedback` requires a boolean `agree`; anything else returns 400.

---

## 4. Frontend

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

## 5. Dependencies

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

## 6. Measured performance

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

## 7. Accuracy on the held test images

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

---

## 8. Repository layout

```
frontend/          11 pages + assets (css, js, fonts, icons)
backend/           app.py (267 lines) · inference.py (565 lines)
models/            live ONNX + metadata + YuNet + archive/
training/          Kaggle/Colab notebooks, result graphs, test media
scripts/           ds.js (server control) · export_onnx.py (.pth → ONNX)
docs/              this file, MODEL_CARD.md, KNOWN_ISSUES.md, DOCUMENTATION.md
uploads/ data/     runtime only, gitignored
venv/              local environment, gitignored
```

**Operating the server:** `npm start` · `npm run status` · `npm run stop` ·
`npm run restart`, or the `START-/STOP-DeepShield.bat` files.
