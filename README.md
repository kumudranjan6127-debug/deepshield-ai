# DeepShield AI — Frontend

Premium, lightweight frontend for **DeepShield AI**, an AI-powered deepfake detection system.
Pure **HTML + CSS + vanilla JavaScript** — no frameworks, no build step, fully functional offline.

## Run it

**Full app (frontend + API)** — Flask serves everything:

```
npm run backend                (= venv\Scripts\python backend/app.py)
→ http://localhost:5000
```

**Frontend only** (no Python needed):

```
npm run dev                    → http://localhost:8000
```

First-time backend setup: `python -m venv venv` then `venv\Scripts\pip install -r requirements.txt`.

The V1 frontend ships with a **simulated analysis engine** (`assets/js/api.js`) so every
flow — upload → processing → result → report — is fully demonstrable without a server.
When the Flask backend lands, point `DS.api` at it (`MODE: 'live'`); no page code changes.

> Demo tip: filenames containing `fake`/`synth` are flagged as deepfakes, `real`/`orig`
> as authentic; anything else gets a deterministic seeded verdict.

## Pages

| Page | File |
|---|---|
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
  index.html ...     10 pages (splash - login - dashboard - ... - report)
  assets/
    css/             fonts, variables (design tokens), base, components, pages/
    js/              utils (DS namespace), api (engine interface), components, pages/
    fonts/ icons/    self-hosted variable fonts, favicon
backend/             Flask server
  app.py             serves frontend + /api/upload + /api/analyze + /api/health
  inference.py       MobileNetV3 engine: YuNet face-crop, image + 1fps video scoring
models/              trained checkpoint (.pth) + YuNet face detector (.onnx)
training/            Colab notebook + result graphs
docs/                full technical documentation
uploads/             temporary staging (auto-cleaned, gitignored)
```

## Design system

- Palette: `#050816` bg · `#0B1221` surface · `#101827` cards · `#3B82F6` primary
- 8px spacing scale, 20px card radius, thin borders, soft shadows
- Type: Space Grotesk (headings) · Inter (body) · JetBrains Mono (all numbers)
- Motion: subtle, `prefers-reduced-motion` respected + in-app toggle

Optimized for CPU-only, low-end hardware (Intel i3 / 8 GB RAM target).
