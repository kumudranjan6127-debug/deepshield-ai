/* ============================================================
   DeepShield AI — settings.js
   Hydrates controls from DS.settings, persists changes with a
   debounced "Settings saved" toast, clear-history modal flow,
   and mock update check.
   ============================================================ */

document.addEventListener('DOMContentLoaded', () => {
  const user = DS.auth.guard();
  if (!user) return;

  /* ---- Elements ---- */
  const frameRate     = document.getElementById('frame-rate');
  const threshold     = document.getElementById('threshold');
  const thresholdVal  = document.getElementById('threshold-val');
  const reducedMotion = document.getElementById('reduced-motion');
  const autoDelete    = document.getElementById('auto-delete');
  const historyCount  = document.getElementById('history-count');
  const historyNoun   = document.getElementById('history-noun');

  /* ---- Hydrate from persisted settings ---- */
  const s = DS.settings.get();
  frameRate.value          = String(s.frameRate);
  threshold.value          = s.threshold;
  thresholdVal.textContent = `${s.threshold}%`;
  reducedMotion.checked    = !!s.reducedMotion;
  autoDelete.checked       = !!s.autoDelete;

  /* ---- Save (persist immediately, debounce the toast) ---- */
  let toastTimer = null;
  function save(patch) {
    DS.settings.set(patch);
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => {
      DS.toast('Settings saved', 'success', { duration: 1800 });
    }, 400);
  }

  frameRate.addEventListener('change', () => {
    save({ frameRate: parseFloat(frameRate.value) });
  });

  threshold.addEventListener('input', () => {
    thresholdVal.textContent = `${threshold.value}%`;
    save({ threshold: parseInt(threshold.value, 10) });
  });

  reducedMotion.addEventListener('change', () => {
    save({ reducedMotion: reducedMotion.checked });
  });

  autoDelete.addEventListener('change', () => {
    save({ autoDelete: autoDelete.checked });
  });

  /* ---- Scan history count + clear flow ---- */
  function renderHistoryCount() {
    const n = DS.history.all().length;
    historyCount.textContent = n;
    historyNoun.textContent = n === 1 ? 'scan' : 'scans';
  }
  renderHistoryCount();

  document.getElementById('clear-confirm').addEventListener('click', () => {
    DS.history.clear();
    renderHistoryCount();
    DS.modal.close('clear-modal');
    DS.toast('Scan history cleared', 'success');
  });

  /* ---- Update check (mock) ---- */
  document.getElementById('check-updates').addEventListener('click', () => {
    DS.toast("You're on the latest version", 'info');
  });
});
