/* ============================================================
   DeepShield AI — result.js
   Reads the completed scan from session storage and renders:
   media preview + meta, model card, verdict hero (confidence
   ring), metrics and action links. No hardcoded verdicts.
   ============================================================ */

document.addEventListener('DOMContentLoaded', () => {
  const user = DS.auth.guard();
  if (!user) return;

  const scan = DS.session.get(DS.KEYS.SCAN);
  if (!scan || !scan.prediction) {
    window.location.replace('dashboard.html');
    return;
  }

  const isFake = scan.prediction === 'deepfake';
  const isVideo = scan.fileType === 'video';
  const reportHref = `report.html?id=${encodeURIComponent(scan.id || '')}`;

  /* ---- Page head ---- */
  document.getElementById('scan-id').textContent = scan.id || '—';
  document.getElementById('scan-date').textContent = DS.util.formatDate(scan.completedAt);
  document.getElementById('report-link').href = reportHref;
  document.getElementById('download-link').href = reportHref;

  renderMedia(scan, isVideo);
  renderMeta(scan, isVideo);
  renderModel();
  renderVerdict(scan, isFake, isVideo);
  renderMetrics(scan);
  renderInsights(scan);
});

/* ---- Analysis insights: judge votes + Grad-CAM heatmap ---- */
function renderInsights(scan) {
  const card = document.getElementById('insights-card');
  const votes = Array.isArray(scan.ensemble)
    ? scan.ensemble.filter(v => typeof v.pFake === 'number')
    : [];
  const explain = scan.explain || null;
  if (!votes.length && !explain) return; // nothing to show (echo/video scans)

  card.hidden = false;

  if (votes.length) {
    document.getElementById('votes-list').innerHTML = votes.map(v => {
      const pct = Math.round(v.pFake * 100);
      return `
        <div class="vote-row">
          <div class="vote-head">
            <span>${DS.util.escapeHtml(v.model)}</span>
            <span class="mono">${pct}% fake</span>
          </div>
          <div class="vote-track">
            <div class="vote-fill${v.pFake >= 0.5 ? ' fake' : ''}" style="width: ${pct}%"></div>
          </div>
        </div>`;
    }).join('');
  }

  if (explain && explain.heatmapDataUrl) {
    document.getElementById('heatmap-img').src = explain.heatmapDataUrl;
    document.getElementById('insight-heat').hidden = false;
  }
  if (explain && explain.note) {
    const note = document.getElementById('focus-note');
    note.textContent = explain.note;
    note.hidden = false;
  }
  if (scan.disputed) document.getElementById('disputed-chip').hidden = false;

  DS.icons();
}

/* ---- Media preview (image or dashed placeholder) ---- */
function renderMedia(scan, isVideo) {
  const wrap = document.getElementById('media-wrap');

  if (scan.previewDataUrl) {
    const img = document.createElement('img');
    img.className = 'result-media';
    img.src = scan.previewDataUrl;
    img.alt = 'Analyzed media preview';
    wrap.appendChild(img);
    return;
  }

  const placeholder = document.createElement('div');
  placeholder.className = 'media-placeholder';
  placeholder.innerHTML = `
    <i data-lucide="${isVideo ? 'file-video' : 'file-image'}" class="icon-xl"></i>
    <span>Preview unavailable</span>
  `;
  wrap.appendChild(placeholder);
  DS.icons();
}

/* ---- File meta rows ---- */
function renderMeta(scan, isVideo) {
  const fileEl = document.getElementById('meta-file');
  const name = scan.fileName || '—';
  fileEl.textContent = DS.util.truncate(name, 32);
  fileEl.title = name;

  document.getElementById('meta-type').textContent = isVideo ? 'Video' : 'Image';
  document.getElementById('meta-size').textContent = DS.util.formatBytes(scan.fileSize);
  document.getElementById('meta-source').textContent =
    scan.source === 'url' ? 'URL' : 'Direct upload';
  document.getElementById('meta-analyzed').textContent = DS.util.formatDate(scan.completedAt);
}

/* ---- Model card (from DS.api.MODEL) ---- */
function renderModel() {
  const m = DS.api.MODEL;
  document.getElementById('model-list').innerHTML = `
    <div class="meta-row"><dt>Model</dt><dd class="mono">${DS.util.escapeHtml(m.name)}</dd></div>
    <div class="meta-row"><dt>Backend</dt><dd>${DS.util.escapeHtml(m.backend)}</dd></div>
    <div class="meta-row"><dt>Parameters</dt><dd class="mono">${DS.util.escapeHtml(m.params)}</dd></div>
    <div class="meta-row"><dt>Input size</dt><dd class="mono">${DS.util.escapeHtml(m.input)}</dd></div>
    <div class="meta-row"><dt>Device</dt><dd class="mono">${DS.util.escapeHtml(m.device)}</dd></div>
  `;
}

/* ---- Verdict hero: ring, badge, risk chip, explanation ---- */
function renderVerdict(scan, isFake, isVideo) {
  const confidence = Math.max(0, Math.min(100, Math.round(scan.confidence || 0)));

  /* Confidence ring — sweep animated via CSS transition on dashoffset */
  const ring = document.getElementById('ring-value');
  const C = 2 * Math.PI * 80; // r = 80 in the 180×180 viewBox
  ring.classList.add(isFake ? 'fake' : 'real');
  ring.style.strokeDasharray = String(C);
  ring.style.strokeDashoffset = String(C);
  document.getElementById('conf-ring')
    .setAttribute('aria-label', `Confidence ${confidence}%`);

  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      ring.style.strokeDashoffset = String(C * (1 - confidence / 100));
    });
  });

  animatePercent(document.getElementById('conf-pct'), confidence);

  /* Verdict badge */
  const badge = document.getElementById('verdict-badge');
  badge.classList.add(isFake ? 'badge-danger' : 'badge-success');
  badge.innerHTML = `
    <i data-lucide="${isFake ? 'alert-triangle' : 'shield-check'}" class="icon"></i>
    ${isFake ? 'Likely Deepfake' : 'Likely Real'}
  `;

  /* Risk chip */
  const risk = scan.riskLevel || (isFake ? 'High' : 'Low');
  const riskClass = { Low: 'badge-success', Medium: 'badge-warning', High: 'badge-danger' }[risk]
    || 'badge-warning';
  const chip = document.getElementById('risk-chip');
  chip.classList.add(riskClass);
  chip.textContent = `Risk level: ${risk}`;

  /* One-sentence explanation, qualitative confidence */
  const qual = confidence >= 85 ? 'high' : 'moderate';
  const unit = isVideo ? 'frames' : 'regions';
  document.getElementById('verdict-note').textContent = isFake
    ? `The model detected patterns consistent with synthetic manipulation across the analyzed ${unit}, with ${qual} confidence.`
    : `No significant manipulation artifacts were detected across the analyzed ${unit} — the model reports ${qual} confidence in this verdict.`;

  DS.icons();
}

/* ---- Metrics row ---- */
function renderMetrics(scan) {
  document.getElementById('metric-time').textContent =
    DS.util.formatDuration(scan.processingTime);
  document.getElementById('metric-frames').textContent =
    scan.framesAnalyzed != null ? String(scan.framesAnalyzed) : '1';
  document.getElementById('metric-device').textContent = scan.device || 'CPU';
}

/* Count-up for the big percentage (respects reduced motion) */
function animatePercent(el, target) {
  const reduced = document.documentElement.dataset.reducedMotion === 'true'
    || window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  if (reduced || target === 0) { el.textContent = `${target}%`; return; }

  const duration = 1000;
  const start = performance.now();
  (function frame(now) {
    const t = Math.min(1, (now - start) / duration);
    const eased = 1 - Math.pow(1 - t, 3);
    el.textContent = `${Math.round(eased * target)}%`;
    if (t < 1) requestAnimationFrame(frame);
  })(start);
}
