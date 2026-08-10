# Architecture

A single-process Flask app that serves a static frontend and runs a
MobileNetV3 classifier on CPU. No database, no queue, no cloud service, no
GPU. That is a constraint, not an omission: the target machine is an i3 with
8 GB of RAM, and the whole backend is 197 MB.

---

## 1. The shape

```
browser ──── static files ──────────────► frontend/  (no build step)
   │
   └──── /api/* ──► app.py ──► security.py   guards
                        │  └──► errors.py     one failure shape
                        │  └──► config.py     every path and limit
                        └────► inference.py   the model
                                   │
                                   ├─ deepshield.onnx        the classifier
                                   └─ face_detection_yunet   the face finder
```

Five backend modules, ~1,400 lines. Each has one job:

| Module | Responsibility | Rule it holds |
|---|---|---|
| `config.py` | Every path, limit and environment switch | Imports nothing from the app, so it can be read on its own |
| `errors.py` | One failure shape, raised as named helpers | `error` stays a plain string — the frontend renders it directly |
| `security.py` | Uploads, SSRF, rate limit, concurrency, cleanup, headers | Refuses; never decides what a thing *is* |
| `inference.py` | Face detection, preprocessing, the model, explanation, video | The only place a model fact exists |
| `app.py` | Routing: validate → infer → respond | Contains no model constants at all |

`app.py` having no model constants is load-bearing. Before Phase 1 it had a
`MODEL_INFO` dict, and it reported MobileNetV3-Small / 2.5M / PyTorch for a
Large ONNX model for weeks. Identity now flows one way only:

```
models/deepshield.onnx.json → engine_info() → /api/health → the browser
```

`tests/test_model_parity.py` asserts the file, the loaded engine and the API
response agree, so that drift cannot happen quietly again.

---

## 2. One image, end to end

```
POST /api/upload          size → extension → MIME → magic bytes → decode
   ↓                      → dimensions; staged under a random id
POST /api/analyze
   ↓
rate limit (5/min)  →  concurrency gate (2 at once, else 503)
   ↓
capped at 1024px longest side            ← a 2687px portrait scored 0.94 fake
   ↓
YuNet face detection → largest face, 0.35 margin crop
   ↓                    (no face → the whole frame, silently — see LIMITATIONS)
JPEG q88 round trip                      ← a pristine camera original scored 0.95
   ↓
resize 224×224 → /255 → ImageNet normalise → CHW
   ↓
forward ×2 (image + mirror), averaged    ← test-time augmentation
   ↓
verdict + confidence + certainty band + risk
   ↓
occlusion sensitivity: 36 patches blanked in turn  ← 93% of the latency
   ↓
staged file deleted
```

The two capped/normalised steps exist because of specific measured false
positives, not on principle. Both are in `MODEL_CARD.md` with the numbers.

---

## 3. Runtime selection

`inference.py` picks a backend at load:

1. **ONNX** when `deepshield.onnx` and its `.json` both exist — run through
   `cv2.dnn`. No PyTorch anywhere. *This is the active path.*
2. **PyTorch** only when no ONNX export exists and torch is importable.
3. Neither → `engine_available()` is false, the app serves an openly
   labelled demo engine and every verdict carries `engine: "simulated"`.

The two backends agree to **3.24e-07** on output probabilities, so which one
is active never changes a verdict. That is re-checked on every export and
asserted by the test suite.

Moving to ONNX took the backend from 1.9 GB to 197 MB. PyTorch was 1.7 GB of
that, to run a 17 MB model.

---

## 4. Video

The classifier is an image model; video is sampled at ~1 frame per second,
60 frames maximum.

```
per-frame P(fake)
   ├── median  0.40    the typical frame; outliers cannot move it
   ├── mean    0.25    the overall level
   └── top-k   0.35    k = 15% of frames — localised manipulation counts,
                       but no single frame decides
            ↓
      combined score
```

Averaging alone dilutes a clip that is only partly manipulated; a maximum
lets one blurred frame accuse an authentic video. `aggregate_frames` is a
pure function — a list of floats in, a dict out — which is why its behaviour
can be pinned against sequences whose right answer is obvious by
construction.

Four temporal signals (face position, face size, landmark jitter, appearance
continuity) come free from data each frame already produced. **None of them
touches the verdict**, because nothing has established what value of any of
them means manipulation.

Frames that will not be sampled are advanced with `grab()` rather than
decoded with `read()` — on a 720p 30 fps clip that is 49% less work walking
the same frames.

---

## 5. Frontend

Eleven pages of plain HTML, CSS and JavaScript. No framework, no bundler, no
`node_modules` at runtime — Node is used only to start the server and serve
static files during development.

```
assets/js/utils.js       DS namespace, storage, formatting
assets/js/api.js         the seam: live backend or simulated engine
assets/js/components.js  shell hydration, engine badge, toasts
assets/js/boot.js        theme and motion preferences before first paint
assets/js/pages/*.js     one file per page
assets/css/variables.css design tokens — every colour and space
```

The browser holds **no facts about the model and no thresholds.** Model
identity and the certainty bands both arrive from `/api/health`; when the
backend is unreachable the UI shows placeholders rather than a remembered
answer. This was not true until Phase 8: eight pages carried
`MobileNetV3-Small / 2.5M / PyTorch` as fallback text long after the backend
stopped saying it.

---

## 6. Trust boundaries

```
┌─ untrusted ────────────────────────────────────────────┐
│  uploaded bytes · uploadId · URLs · filenames · fields  │
└────────────────────────┬───────────────────────────────┘
                         │  security.py validates
┌────────────────────────▼───────────────────────────────┐
│  trusted: decoded media, resolved paths, model output   │
└─────────────────────────────────────────────────────────┘
```

`frontend/` is the only directory served. Backend code, models, uploads and
`eval_data/` all live outside that root. See `SECURITY.md`.

---

## 7. What runs where

| Concern | Where | Why not elsewhere |
|---|---|---|
| Verdict | Backend | The browser must not be able to claim a result |
| Certainty band | Backend, published to the browser | One threshold table, not two |
| Risk label | Backend | Same reason |
| History | Browser `localStorage` | No account system, and media should not be kept |
| Preview thumbnails | Browser | Media never has to leave the machine for the demo path |
| Feedback | Backend, `data/feedback.jsonl` | An evaluation signal — never a training label; nothing reaches the model automatically |

---

## 8. Testing and evaluation

Two different questions, two different tools.

```
python -m pytest                      is it correct?      261 tests, ~20 s
python scripts/regression_test.py     did anything change? 24 recorded values
python scripts/evaluate.py            how accurate is it?  needs labelled data
python scripts/benchmark.py           how fast is it?      stage by stage
```

The suite runs against Flask's test client — no server, no network, and test
media is generated from repository material rather than committed.

`scripts/ds_metrics.py` is the only implementation of the evaluation
arithmetic in the project. The training notebook computes no metrics; it
writes raw scores, and `evaluate.py --from-csv` turns them into the table. A
Kaggle number and a local number are therefore the same code.

---

## 9. Deliberate omissions

| Not here | Why |
|---|---|
| Database | Nothing needs to persist. History is per-browser |
| Queue / workers | One CPU-bound analysis at a time is the honest capacity |
| Docker | The target was "runs on a laptop with two commands" |
| GPU | 4 cores and 260 MB is the whole budget |
| Framework on the frontend | Eleven pages do not need a build step |
| Server-side sessions | There is no authorisation model to protect |

Each of these is a decision that could be revisited. None of them is an
oversight.
