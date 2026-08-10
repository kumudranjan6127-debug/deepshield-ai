/* ============================================================
   DeepShield AI — dashboard.js
   Greeting, stats, recent scans table, mock system metrics.
   ============================================================ */

document.addEventListener('DOMContentLoaded', () => {
  const user = DS.auth.guard();
  if (!user) return;

  /* ---- Greeting ---- */
  const greeting = document.getElementById('greeting');
  const firstName = String(user.name || '').split(' ')[0];
  greeting.textContent = `${DS.util.greeting()}, ${firstName}`;

  /* ---- Stats + recent scans ---- */
  const history = DS.history.all();

  const fakes = history.filter(s => s.prediction === 'deepfake').length;
  const reals = history.filter(s => s.prediction === 'real').length;
  const avgConf = history.length
    ? Math.round(history.reduce((sum, s) => sum + (s.confidence || 0), 0) / history.length)
    : null;

  animateCount(document.getElementById('stat-total'), history.length);
  animateCount(document.getElementById('stat-fake'), fakes);
  animateCount(document.getElementById('stat-real'), reals);
  document.getElementById('stat-conf').textContent = avgConf === null ? '—' : `${avgConf}%`;

  renderRecent(history);

  /* ---- Real engine status from the backend ---- */
  hydrateEngineStatus();
  renderFeedbackStat();
});

/* Real-world accuracy as rated by the user (from result-page feedback) */
function renderFeedbackStat() {
  // Skipped ("not sure") entries carry no verdict — they must not count
  const fb = DS.store.get('ds_feedback', []).filter(f => typeof f.agree === 'boolean');
  if (!fb.length) return;
  const correct = fb.filter(f => f.agree).length;
  document.getElementById('fb-val').textContent =
    `${Math.round((correct / fb.length) * 100)}% (${correct}/${fb.length})`;
  document.getElementById('fb-row').hidden = false;
}

/* Count-up animation for stat values (respects reduced motion) */
function animateCount(el, target) {
  const reduced = document.documentElement.dataset.reducedMotion === 'true'
    || window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  if (reduced || target === 0) { el.textContent = target; return; }

  const duration = 700;
  const start = performance.now();
  (function frame(now) {
    const t = Math.min(1, (now - start) / duration);
    const eased = 1 - Math.pow(1 - t, 3);
    el.textContent = Math.round(eased * target);
    if (t < 1) requestAnimationFrame(frame);
  })(start);
}

function renderRecent(history) {
  const body = document.getElementById('recent-body');
  const table = document.getElementById('recent-table');
  const empty = document.getElementById('recent-empty');

  if (!history.length) {
    table.hidden = true;
    empty.hidden = false;
    DS.icons();
    return;
  }

  const rows = history.slice(0, 6).map(scan => {
    const isFake = scan.prediction === 'deepfake';
    return `
      <tr>
        <td title="${DS.util.escapeHtml(scan.fileName)}">${DS.util.truncate(DS.util.escapeHtml(scan.fileName), 28)}</td>
        <td>${scan.fileType === 'video' ? 'Video' : 'Image'}</td>
        <td>
          <span class="badge ${isFake ? 'badge-danger' : 'badge-success'}">
            <span class="badge-dot"></span>${isFake ? 'Likely Deepfake' : 'Likely Real'}
          </span>
        </td>
        <td class="mono">${scan.confidence}%</td>
        <td>${DS.util.formatDate(scan.completedAt)}</td>
        <td>
          <a class="btn btn-ghost btn-sm" href="report.html?id=${encodeURIComponent(scan.id)}">Report</a>
        </td>
      </tr>`;
  }).join('');

  body.innerHTML = rows;
}

/* Ask the backend which engine is running. Live engine (real model
   loaded) → green badge + real test accuracy. No backend / no model
   → the honest "Simulated (demo)" default stays. */
async function hydrateEngineStatus() {
  const badge = document.getElementById('engine-badge');
  if (!badge) return;

  const health = await DS.server.health(); // shared with DS.server.hydrate()
  if (!health) return;                    // no backend — the demo badge stands

  // One painter for every engine badge in the app; see components.js.
  DS.server.paintBadge(badge, health.engine === 'live' ? 'live' : 'simulated');
  if (health.engine !== 'live') return;

  if (health.test_accuracy != null) {
    document.getElementById('acc-val').textContent = `${health.test_accuracy}%`;
    document.getElementById('acc-row').hidden = false;
  }
}
