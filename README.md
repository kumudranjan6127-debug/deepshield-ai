# DeepShield AI

Deepfake detection for images and video — a trained MobileNetV3 plus two
verifier models, running entirely on CPU. Flask backend, framework-free
frontend, no cloud.

## Run it

**Start the app** (backend + UI, on http://localhost:5000):

```
npm start          # start the backend and report what it is running
npm run status     # is it up? which model? which accuracy?
npm run stop       # stop it and free the port
npm run restart    # stop, then start
```

`npm start` refuses to leave duplicates behind: it clears the port first,
waits for the model to load, then prints the engine, checkpoint and
accuracies. `npm run stop` kills the whole process tree.

Not a terminal person? Double-click **START-DeepShield.bat** (it opens the
browser too) and **STOP-DeepShield.bat** when finished.

**Frontend only** (no Python needed — the app runs its simulated engine and
labels itself "Simulated (demo)"): `npm run dev` → http://localhost:8000

First-time backend setup:

```
python -m venv venv
venv\Scripts\pip install -r requirements.txt
```

Server output goes to `backend.log`. Editing backend code? Set `DS_DEBUG=1`
for auto-reload — off by default, because its reloader spawns a second
process that respawns the first.

> Demo tip: with no model in `models/`, the app runs a simulated engine and
> says so on the dashboard. Filenames containing `fake` are flagged, `real`
> are cleared — useful for walking through the UI without inference.

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
| v1 | — | — | — | 96.94% test; learned the dataset's pipeline fingerprint |
| v2 | 99.40% | 98.54% | — | anti-shortcut augmentation, robust-selected |
| **v3 (live)** | **99.90%** | **99.18%** | **100.00%** | multi-generator: StyleGAN1 + StyleGAN2 + diffusion |

Every version is kept in `models/archive/` — rolling back is one file copy
(`models/archive/README.md` explains what each version taught us).

At inference the image is capped at 1024px, the face is cropped with YuNet,
and three models vote: ours leads, two pretrained verifiers advise. A
Grad-CAM heatmap shows where the model actually looked.

## Design system

- Palette: `#050816` bg · `#0B1221` surface · `#101827` cards · `#3B82F6` primary
- 8px spacing scale, 20px card radius, glass surfaces with a two-tier
  effects system (full blur / lite — switchable in Settings)
- Type: Space Grotesk (headings) · Inter (body) · JetBrains Mono (numbers)
- Motion: subtle, `prefers-reduced-motion` respected + in-app toggle

Optimized for CPU-only, low-end hardware (Intel i3 / 8 GB RAM target).
