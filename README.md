# DeepShield AI

Deepfake detection for images and video — a MobileNetV3 we trained ourselves
plus two verifier models, running entirely on CPU. Flask backend,
framework-free frontend, no cloud, no GPU.

The trained model ships **inside this repository**, so a fresh clone works
after `pip install` — nothing to download separately.

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

Roughly 1 GB of downloads, mostly PyTorch — one coffee. On Linux, install
torch from the CPU index first (see the note at the top of
`requirements.txt`) or you will pull the 2 GB CUDA build for nothing. On
Windows, if importing torch later fails with a DLL error, install the
[Visual C++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe).

### 4. Run it

```
npm start
```

Open **http://localhost:5000**. Log in with any email and password, or use
**Continue as guest**.

> The first image you analyze downloads the two verifier models (~930 MB,
> cached once). Skip them entirely by running offline — the app drops to our
> own model and keeps working.

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
     model     mobilenet_v3_large  deepshield_mobilenetv3.pth
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

## Troubleshooting

| Symptom | Fix |
|---|---|
| `EADDRINUSE` / port 5000 busy | `npm run stop`, then `npm start` |
| `npm start` says "did not come up" | Read `backend.log` — the real error is there |
| `python` not recognised | Reinstall with **Add to PATH** ticked, open a new terminal |
| Torch import fails with a DLL error (Windows) | Install the [VC++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe) |
| Verifier model download stalls | `set HF_HUB_DISABLE_XET=1` before starting — Hugging Face's Xet transfer fails on some networks; plain HTTP works |
| Dashboard says "Simulated (demo)" | `models/deepshield_mobilenetv3.pth` is missing, or torch is not installed |
| Analysis is slow the first time | Models load on first use (~10 s); afterwards a photo takes well under a second |

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
  inference.py       MobileNetV3 engine: YuNet face-crop, ensemble, Grad-CAM
models/              live checkpoint + YuNet detector; archive/ keeps every version
training/            Kaggle/Colab notebooks + result graphs
scripts/ds.js        server control (start/stop/status/restart)
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
and three models vote: ours leads, two pretrained verifiers advise. A
Grad-CAM heatmap shows where the model actually looked.

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
