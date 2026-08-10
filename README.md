# DeepShield AI

Deepfake detection for images and video — a MobileNetV3 we trained ourselves,
running entirely on CPU. Flask backend, framework-free frontend, no cloud,
no GPU.

The trained model ships **inside this repository** as ONNX, so a fresh clone
works after `pip install` — no PyTorch, no downloads, ~160 MB of
dependencies in total.

---

## Setup on a new machine

### 1. Install the prerequisites

| Need | Version | Where |
|---|---|---|
| **Python** | 3.10 – 3.14 | [python.org/downloads](https://www.python.org/downloads/) — tick **"Add python.exe to PATH"** during install |
| **Node.js** | 18+ | [nodejs.org](https://nodejs.org/) — used only for the control script |
| **Git** | any | [git-scm.com](https://git-scm.com/) |

Check they work (open a **new** terminal after installing):

```
python --version
node --version
```

> **Windows note:** if `python` opens the Microsoft Store, turn off the fake
> shortcut: *Settings → Apps → Advanced app settings → App execution
> aliases* → switch off **python.exe** and **python3.exe**.

### 2. Get the code

```
git clone https://github.com/kumudranjan6127-debug/deepshield-ai.git
cd deepshield-ai
```

### 3. Create the environment and install

**Windows**
```
python -m venv venv
venv\Scripts\pip install -r requirements.txt
```

**macOS / Linux**
```
python3 -m venv venv
venv/bin/pip install -r requirements.txt
```

About 160 MB — Flask, OpenCV, Pillow and NumPy. The model runs as ONNX
through OpenCV, so PyTorch is only needed for training or for exporting a
new model (see the optional extras in `requirements.txt`).

### 4. Run it

```
npm start
```

Open **http://localhost:5000**. Log in with any email and password, or use
**Continue as guest**. That is the whole setup — the model is already there.

> **Want the ensemble?** Two pretrained verifier models can cross-check every
> verdict. They are off by default (~1 GB of downloads and RAM, and our own
> model scores every image in the held set correctly alone). To enable:
> `pip install transformers torch`, then start with `DS_VERIFIERS=1`.

---

## Everyday commands

```
npm start          # start the backend and report what it is running
npm run status     # is it up? which model? which accuracy?
npm run stop       # stop it and free the port
npm run restart    # stop, then start
```

`npm start` prints exactly what you are running:

```
  ● DeepShield backend is running   http://localhost:5000
     engine    live — real model
     model     mobilenet_v3_large  deepshield.onnx
     accuracy  val 99.9% · TPDN 100%
     processes 1
```

`npm start` clears the port before binding, so repeated starts never leave
duplicate servers behind, and `npm run stop` terminates the whole process
tree — a session once accumulated twelve Flask instances because debug
mode's reloader respawns its own child.

**Prefer clicking?** Double-click **START-DeepShield.bat** (it opens the
browser too) and **STOP-DeepShield.bat** when finished. On macOS/Linux the
npm commands work the same way.

**Frontend only** — no Python needed. The app detects that no backend is
answering, runs its simulated engine and labels itself "Simulated (demo)":

```
npm run dev        # → http://localhost:8000
```

Server output goes to `backend.log`. Editing backend code? Set `DS_DEBUG=1`
for auto-reload — off by default, because its reloader spawns a second
process that respawns the first.

---

## Running the tests

```
pip install pytest
python -m pytest
```

That is the whole command. 241 tests, about 20 seconds, and it needs
**no running server and no network** — it drives Flask's test client
in-process and stubs DNS where the SSRF checks need to resolve a name.

```
python -m pytest -m security      # one category
python -m pytest -m "not slow"    # skip anything that loads the model
python -m pytest -q tests/test_upload.py
```

| Category | Covers |
|---|---|
| `api` | endpoint contracts and the shape of every response |
| `upload` | an extension is not evidence: MIME, magic bytes, decoder, dimensions |
| `validation` | fields, URLs, media, and the error-code vocabulary |
| `security` | SSRF, path traversal, rate limiting, concurrency, cleanup |
| `inference` | real vs generated, no face, two faces, tiny face, 3000px, heavy compression |
| `video` | frame aggregation, temporal signals, the 60-frame cap, broken clips |
| `parity` | ONNX and PyTorch agree; the model reports what it is |
| `metrics` | the evaluation arithmetic, against hand-computed answers |

Test images and clips are **generated, not committed** — `tests/conftest.py`
builds them from the sample faces already in `training/`, so a fresh clone
has full coverage without carrying binaries.

Separately, `python scripts/regression_test.py verify` answers a different
question — *did this change alter any behaviour?* — by diffing against a
recorded baseline. That one does need the server running:

```
DS_RATE_LIMIT=50 npm start
DS_RATE_LIMIT=50 python scripts/regression_test.py verify
```

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `EADDRINUSE` / port 5000 busy | `npm run stop`, then `npm start` |
| `npm start` says "did not come up" | Read `backend.log` — the real error is there |
| `python` not recognised | Reinstall with **Add to PATH** ticked, open a new terminal |
| Dashboard says "Simulated (demo)" | `models/deepshield.onnx` (+ its `.json`) is missing |
| Verifier model download stalls | `set HF_HUB_DISABLE_XET=1` before starting — Hugging Face's Xet transfer fails on some networks; plain HTTP works |
| Torch import fails with a DLL error (Windows) | Only affects the optional extras — install the [VC++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe) |
| Analysis feels slow | ~0.6 s per image including the heatmap; the first request also loads the model |

---

## Pages

| Page | File |
|---|---|
| Landing | `landing.html` (served at `/`) |
| Splash | `index.html` |
| Login | `login.html` |
| Dashboard | `dashboard.html` |
| Upload Image / Video | `upload-image.html`, `upload-video.html` |
| Processing | `processing.html` |
| Result | `result.html` |
| Report (print → PDF) | `report.html` |
| About / Settings | `about.html`, `settings.html` |

## Structure

```
frontend/            UI — one .html per page + assets/
  landing.html       public front door; index.html … the 10 app pages
  assets/
    css/             fonts, variables (design tokens), base, components, pages/
    js/              utils (DS namespace), api (engine interface), components, pages/
    fonts/ icons/    self-hosted variable fonts, favicon
backend/             Flask server
  app.py             serves frontend + /api/upload + /api/analyze + /api/health
  inference.py       engine: YuNet face-crop, ONNX/torch backends, saliency
models/              deepshield.onnx (+ .json) · YuNet · archive/ every version
training/            Kaggle/Colab notebooks + result graphs
scripts/ds.js        server control (start/stop/status/restart)
scripts/export_onnx.py   .pth → single-file ONNX, with a parity check
docs/                full technical documentation
uploads/ data/       runtime only (auto-cleaned, gitignored)
```

## The model

| Version | Val | Robust | TPDN holdout | Notes |
|---|---|---|---|---|
| v1 | — | — | — | 96.94% test; had learned the dataset's pipeline fingerprint |
| v2 | 99.40% | 98.54% | — | anti-shortcut augmentation, robust-selected |
| **v3 (live)** | **99.90%** | **99.18%** | **100.00%** | multi-generator: StyleGAN1 + StyleGAN2 + diffusion |

Every version stays in `models/archive/` — rolling back is one file copy, and
`models/archive/README.md` records what each version taught us.

At inference the image is capped at 1024px, the face is cropped with YuNet,
and the classifier runs as ONNX through OpenCV — the same weights, verified
to agree with PyTorch to ~1e-7 on every probability. Optional verifiers can
cross-check the verdict.

The heatmap comes from **occlusion sensitivity**: patches are blanked out one
at a time and the regions whose removal moves the score the most are the ones
the model relied on. It needs only forward passes, so it works on both
backends — and it measures the model's dependence rather than interpreting
its internals.

**Scope, honestly:** it detects fully AI-generated faces well (StyleGAN,
diffusion). Face-swap video deepfakes are a different artefact family and are
not covered yet. Results are a strong signal, not forensic proof.

## Design system

- Palette: `#050816` bg · `#0B1221` surface · `#101827` cards · `#3B82F6` primary
- 8px spacing scale, 20px card radius, glass surfaces with a two-tier effects
  system (full blur / lite — switchable in Settings)
- Type: Space Grotesk (headings) · Inter (body) · JetBrains Mono (numbers)
- Motion: subtle, `prefers-reduced-motion` respected + in-app toggle

Optimized for CPU-only, low-end hardware (Intel i3 / 8 GB RAM target).
