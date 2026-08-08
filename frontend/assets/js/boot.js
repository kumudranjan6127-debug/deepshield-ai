/* ============================================================
   DeepShield AI — boot.js  (loaded BLOCKING in <head>, pre-paint)
   Decides two things before the first paint, so nothing flashes:
     1. entry animations — only on the session's first page
     2. visual effects tier — 'full' (real glass) or 'lite'
   Must stay tiny; it reads storage directly instead of waiting for DS.
   ============================================================ */
(function () {
  var root = document.documentElement;

  /* ---- 1. Animate only the first page of the session ---- */
  try {
    if (sessionStorage.getItem('ds_nav')) root.classList.add('instant');
    else sessionStorage.setItem('ds_nav', '1');
  } catch (e) { /* storage blocked — animate every time, no harm */ }

  /* ---- 2. Visual effects tier ---- */
  var pref = 'auto';
  try {
    var s = JSON.parse(localStorage.getItem('ds_settings') || '{}');
    if (s.effects) pref = s.effects;
  } catch (e) { /* ignore malformed settings */ }

  var fx;
  if (pref === 'full' || pref === 'lite') {
    fx = pref;
  } else {
    // Auto: backdrop-filter is expensive on low-core / low-memory machines
    var cores = navigator.hardwareConcurrency || 4;
    var mem = navigator.deviceMemory || 4;
    var capable = cores >= 8 && mem >= 8 &&
      window.matchMedia('(hover: hover)').matches &&
      !window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    fx = capable ? 'full' : 'lite';
  }
  root.dataset.fx = fx;
})();
