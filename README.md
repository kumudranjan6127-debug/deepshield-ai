# DeepShield AI — Frontend

Premium, lightweight frontend for **DeepShield AI**, an AI-powered deepfake detection system.
Pure **HTML + CSS + vanilla JavaScript** — no frameworks, no build step, fully functional offline.

## Run it

**Full app (frontend + API)** — Flask serves everything:

```
venv\Scripts\python app.py     (or: npm run backend)
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
assets/
  css/
    fonts.css        self-hosted variable fonts (Inter, Space Grotesk, JetBrains Mono)
    variables.css    design tokens — single source of truth
    base.css         reset, typography, utilities
    components.css   component library (shell, buttons, cards, forms, …)
    pages/           one stylesheet per page
  js/
    utils.js         DS namespace: storage, history, settings, auth, formatters
    api.js           analysis engine interface (simulated ⇄ Flask-ready)
    components.js    toasts, modals, sidebar, shell hydration, Lucide icons
    pages/           one script per page
    vendor/          lucide.min.js (icons, vendored — works offline)
  fonts/  icons/  images/
*.html               one file per page
```

## Design system

- Palette: `#050816` bg · `#0B1221` surface · `#101827` cards · `#3B82F6` primary
- 8px spacing scale, 20px card radius, thin borders, soft shadows
- Type: Space Grotesk (headings) · Inter (body) · JetBrains Mono (all numbers)
- Motion: subtle, `prefers-reduced-motion` respected + in-app toggle

Optimized for CPU-only, low-end hardware (Intel i3 / 8 GB RAM target).
