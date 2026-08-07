/* ============================================================
   DeepShield AI — splash.js
   Boot sequence: real progress bar + %, engine log lines that
   light up, then route (dashboard if signed in, else login).
   Click / Enter skips. Reduced motion → quick fade, no theater.
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

  /* ---- Reduced motion: no theater, quick exit ---- */
  if (reduced) {
    bar.style.width = '100%';
    pct.textContent = '100%';
    setTimeout(go, 800);
    return;
  }

  /* ---- Boot log lines (show at %, tick done at %) ---- */
  const STEPS = [
    { text: 'engine  · loading MobileNetV3-Small', showAt: 8,  doneAt: 38 },
    { text: 'device  · CPU inference ready',       showAt: 40, doneAt: 68 },
    { text: 'session · local & private',           showAt: 70, doneAt: 92 },
  ];
  STEPS.forEach(step => {
    const line = document.createElement('div');
    line.className = 'boot-line';
    line.innerHTML = `<span class="tick">✓</span><span>${step.text}</span>`;
    log.appendChild(line);
    step.el = line;
  });

  /* ---- Drive progress with rAF (2.6s total) ---- */
  const TOTAL = 2600;
  const start = performance.now();

  (function frame(now) {
    if (gone) return;
    const p = Math.min(100, ((now - start) / TOTAL) * 100);

    bar.style.width = `${p}%`;
    pct.textContent = `${Math.round(p)}%`;

    STEPS.forEach(step => {
      if (p >= step.showAt) step.el.classList.add('show');
      if (p >= step.doneAt) step.el.classList.add('done');
    });

    if (p >= 100) { setTimeout(go, 250); return; }
    requestAnimationFrame(frame);
  })(start);
});
