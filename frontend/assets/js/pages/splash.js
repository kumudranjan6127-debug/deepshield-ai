/* ============================================================
   DeepShield AI — splash.js
   Boot sequence. The animation is not filler: it runs the same
   stages the analysis pipeline does — detect a face, place the
   five YuNet landmarks, close the 0.35-margin crop, sweep the
   6x6 occlusion grid — so the first screen is a description of
   the app rather than a loading bar with a logo on it.

   ~5.5s, skippable with a click or Enter. Reduced motion gets
   the state without the theatre.
   ============================================================ */

document.addEventListener('DOMContentLoaded', () => {
  const dest = DS.auth.user() ? 'dashboard.html' : 'login.html';

  const reduced =
    document.documentElement.dataset.reducedMotion === 'true' ||
    window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  const bar = document.getElementById('boot-bar');
  const pct = document.getElementById('boot-pct');
  const log = document.getElementById('boot-log');

  /* ---- Redirect (once) ---- */
  let gone = false;
  function go() {
    if (gone) return;
    gone = true;
    window.location.replace(dest);
  }
  document.addEventListener('click', go);
  document.addEventListener('keydown', e => { if (e.key === 'Enter') go(); });

  buildOcclusionGrid();

  /* ---- Reduced motion: no theatre, quick exit ---- */
  if (reduced) {
    bar.style.width = '100%';
    pct.textContent = '100%';
    setTimeout(go, 800);
    return;
  }

  /* ---- Boot log ----
     The engine line names whatever model is actually loaded. It used to
     read "MobileNetV3-Small", which stopped being true when V3 shipped as
     Large — on the first screen anyone sees. It is filled from
     /api/health now, like every other model fact in the app, and shows a
     neutral placeholder until that answers. */
  const STEPS = [
    { key: 'detector', value: 'YuNet face model',  showAt: 6,  doneAt: 30 },
    { key: 'engine',   value: 'loading…',          showAt: 26, doneAt: 58, engine: true },
    { key: 'device',   value: 'CPU inference',     showAt: 54, doneAt: 78 },
    { key: 'session',  value: 'local & private',   showAt: 74, doneAt: 94 },
  ];

  STEPS.forEach(step => {
    const line = document.createElement('div');
    line.className = 'boot-line';

    const tick = document.createElement('span');
    tick.className = 'tick';
    tick.textContent = '✓';

    const key = document.createElement('span');
    key.className = 'boot-key';
    key.textContent = step.key;

    const value = document.createElement('span');
    value.className = 'boot-val';
    value.textContent = step.value;

    line.append(tick, key, value);
    log.appendChild(line);
    step.el = line;
    step.valueEl = value;
  });

  nameTheEngine(STEPS);

  /* ---- Drive the timeline ---- */
  const TOTAL = 5500;
  const start = performance.now();

  (function frame(now) {
    if (gone) return;
    const elapsed = now - start;
    /* Ease the bar so it does not crawl at a constant rate for five
       seconds — it moves with the stages rather than against them. */
    const linear = Math.min(1, elapsed / TOTAL);
    const p = Math.min(100, (1 - Math.pow(1 - linear, 1.7)) * 100);

    bar.style.width = `${p}%`;
    pct.textContent = `${Math.round(p)}%`;

    STEPS.forEach(step => {
      if (p >= step.showAt) step.el.classList.add('show');
      if (p >= step.doneAt) step.el.classList.add('done');
    });

    if (linear >= 1) { setTimeout(go, 350); return; }
    requestAnimationFrame(frame);
  })(start);
});


/* ---- The 6x6 grid the explanation actually uses ----
   Thirty-six cells over the face, each one flashing in turn, which is
   exactly what occlusion sensitivity does: blank a patch, see how far the
   score moves. Built here rather than written out in the markup so the
   grid size stays a single number. */
function buildOcclusionGrid(grid = 6) {
  const host = document.querySelector('.occl');
  if (!host) return;

  const x0 = 46, y0 = 34, size = 108, cell = size / grid;
  const svgNS = 'http://www.w3.org/2000/svg';

  for (let row = 0; row < grid; row++) {
    for (let col = 0; col < grid; col++) {
      const rect = document.createElementNS(svgNS, 'rect');
      rect.setAttribute('x', (x0 + col * cell).toFixed(2));
      rect.setAttribute('y', (y0 + row * cell).toFixed(2));
      rect.setAttribute('width', (cell - 1.5).toFixed(2));
      rect.setAttribute('height', (cell - 1.5).toFixed(2));
      rect.setAttribute('rx', '1');
      /* Diagonal order reads as a sweep rather than a raster scan. */
      rect.style.setProperty('--d', row + col);
      host.appendChild(rect);
    }
  }
}


/* ---- Name the loaded model, or say nothing ----
   Health may be slow, or absent entirely on a static deploy. Either way
   the line stays honest: a neutral placeholder rather than a guess. */
function nameTheEngine(steps) {
  const step = steps.find(s => s.engine);
  if (!step || !DS.server || !DS.server.health) return;

  DS.server.health().then(health => {
    if (!health) return;
    const name = [health.model_name, health.version].filter(Boolean).join(' ')
      || health.architecture;
    if (!name) return;

    const runtime = health.runtime ? ` · ${health.runtime}` : '';
    step.valueEl.textContent = `${name}${runtime}`;
  }).catch(() => { /* no backend — the placeholder stands */ });
}
