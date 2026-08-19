/* DeepShield dashboard: local scan history, feedback and live engine status. */

document.addEventListener('DOMContentLoaded', () => {
  const user = DS.auth.guard();
  if (!user) return;

  const greeting = document.getElementById('greeting');
  const firstName = String(user.name || '').trim().split(/\s+/)[0];
  greeting.textContent = firstName ? `${DS.util.greeting()}, ${firstName}` : DS.util.greeting();

  const history = DS.history.all();
  const conclusive = history.filter(s => !isInconclusive(s));
  const fakes = conclusive.filter(s => s.prediction === 'deepfake').length;
  const reals = conclusive.filter(s => s.prediction === 'real').length;
  const avgConf = conclusive.length
    ? Math.round(conclusive.reduce((sum, s) => sum + (s.confidence || 0), 0) / conclusive.length)
    : null;

  animateCount(document.getElementById('stat-total'), history.length);
  animateCount(document.getElementById('stat-fake'), fakes);
  animateCount(document.getElementById('stat-real'), reals);
  document.getElementById('stat-conf').textContent = avgConf === null ? '—' : `${avgConf}%`;

  renderRecent(history);
  hydrateEngineStatus();
  renderFeedbackStat();
});

function isInconclusive(scan) {
  return scan && (scan.insufficientEvidence === true
    || scan.findingType === 'inconclusive'
    || (Number(scan.confidence) <= 50 && scan.prediction === 'real'));
}

function scanPresentation(scan) {
  if (isInconclusive(scan)) return { cls: 'badge-warning', label: 'Inconclusive' };
  if (scan.prediction === 'deepfake') {
    if (scan.findingType === 'ai_generated') return { cls: 'badge-danger', label: 'Likely AI-generated' };
    if (scan.findingType === 'face_manipulation') return { cls: 'badge-danger', label: 'Likely Manipulated' };
    return { cls: 'badge-danger', label: 'Likely Synthetic' };
  }
  return { cls: 'badge-success', label: 'Likely Real' };
}

function renderFeedbackStat() {
  const fb = DS.store.get(DS.KEYS.FEEDBACK || 'ds_feedback', [])
    .filter(f => typeof f.agree === 'boolean');
  if (!fb.length) return;
  const correct = fb.filter(f => f.agree).length;
  document.getElementById('fb-val').textContent =
    `${Math.round((correct / fb.length) * 100)}% (${correct}/${fb.length})`;
  document.getElementById('fb-row').hidden = false;
}

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

  body.innerHTML = history.slice(0, 6).map(scan => {
    const p = scanPresentation(scan);
    return `
      <tr>
        <td title="${DS.util.escapeHtml(scan.fileName || '')}">${DS.util.truncate(DS.util.escapeHtml(scan.fileName || '—'), 28)}</td>
        <td>${scan.fileType === 'video' ? 'Video' : 'Image'}</td>
        <td><span class="badge ${p.cls}"><span class="badge-dot"></span>${p.label}</span></td>
        <td class="mono">${scan.confidence != null ? `${scan.confidence}%` : '—'}</td>
        <td>${DS.util.formatDate(scan.completedAt)}</td>
        <td><a class="btn btn-ghost btn-sm" href="report.html?id=${encodeURIComponent(scan.id)}">Report</a></td>
      </tr>`;
  }).join('');
}

async function hydrateEngineStatus() {
  const badge = document.getElementById('engine-badge');
  if (!badge) return;
  const health = await DS.server.health();
  if (!health) {
    DS.server.paintBadge(badge, DS.api.isExplicitDemo() ? 'simulated' : 'offline');
    return;
  }
  DS.server.paintBadge(badge, health.engine === 'live' ? 'live' : 'simulated');
  if (health.engine !== 'live') return;
  if (health.test_accuracy != null) {
    document.getElementById('acc-val').textContent = `${health.test_accuracy}%`;
    document.getElementById('acc-row').hidden = false;
  }
}
