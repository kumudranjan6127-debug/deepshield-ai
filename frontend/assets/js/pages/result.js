/* DeepShield result page — renders stored scan data; no hardcoded verdicts. */
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
  renderVideo(scan);
  renderVerdictFacts(scan);
  renderWhy(scan);
  bindFeedback(scan);

  document.addEventListener('ds:server-ready', renderModel);
  document.addEventListener('ds:server-ready', renderCertainty);
  document.addEventListener('ds:server-ready', () => renderVerdictFacts(scan));
});

function originVote(scan) {
  return Array.isArray(scan.ensemble)
    ? scan.ensemble.find(v => v && v.kind === 'ai-origin/full-frame' && typeof v.pFake === 'number')
    : null;
}

function renderInsights(scan) {
  const card = document.getElementById('insights-card');
  const votes = Array.isArray(scan.ensemble)
    ? scan.ensemble.filter(v => typeof v.pFake === 'number')
    : [];
  const explain = scan.explain || null;
  if (!votes.length && !explain) return;

  card.hidden = false;
  if (votes.length) {
    document.getElementById('votes-list').innerHTML = votes.map(v => {
      const pct = Math.round(v.pFake * 100);
      const suffix = v.kind === 'ai-origin/full-frame' ? ' AI-origin score' : ' fake score';
      return `
        <div class="vote-row">
          <div class="vote-head">
            <span>${DS.util.escapeHtml(v.model)}</span>
            <span class="mono">${pct}%${DS.util.escapeHtml(suffix)}</span>
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
  renderScope(scan);
  DS.icons();
}

function renderScope(scan) {
  if (scan.faceFound === undefined) return;
  const box = document.getElementById('scope-note');
  const title = document.getElementById('scope-title');
  const body = document.getElementById('scope-body');
  const faces = Number(scan.facesFound || 0);
  const fullFrame = originVote(scan);

  if (scan.faceFound === false) {
    title.textContent = 'No face detected';
    body.textContent = fullFrame
      ? 'The face detector had no reliable face crop. A separate full-frame AI-origin model still analysed the media; inspect the model scores below because that evidence is independent of the face detector.'
      : 'No reliable face evidence was available. Treat a 50% result as no answer rather than evidence that the media is real.';
  } else if (faces > 1) {
    title.textContent = `${faces} faces detected`;
    body.textContent =
      `Every detected face was analysed and the face-level verdict uses the most suspicious one. ` +
      `${fullFrame ? 'A separate full-frame AI-origin score is also shown below.' : ''}`;
  } else {
    box.hidden = true;
    return;
  }
  box.hidden = false;
}

function renderVideo(scan) {
  const v = scan.video;
  if (!v) return;

  const card = document.getElementById('video-card');
  card.hidden = false;
  const pct = x => (typeof x === 'number' ? `${Math.round(x * 100)}%` : '—');
  const row = (label, value, title) =>
    `<div class="meta-row"${title ? ` title="${DS.util.escapeHtml(title)}"` : ''}>
       <dt>${DS.util.escapeHtml(label)}</dt>
       <dd class="mono">${DS.util.escapeHtml(String(value))}</dd>
     </div>`;

  const suspiciousAt = pct(v.suspiciousAt);
  const rows = [
    row('Frames analyzed', v.framesAnalyzed ?? '—'),
  ];
  if (v.faceFramesAnalyzed != null) {
    rows.push(row('Frames with face evidence', `${v.faceFramesAnalyzed} / ${v.framesAnalyzed ?? '—'}`));
  }
  rows.push(
    row('Suspicious face frames', `${v.suspiciousFrames ?? '—'} / ${v.faceFramesAnalyzed ?? v.framesAnalyzed ?? '—'}`,
        `Face-bearing frames scoring ${suspiciousAt} or higher`),
    row('Peak face fake score', pct(v.peakFakeScore)),
    row('Median face fake score', pct(v.medianFakeScore)),
    row('Mean face fake score', pct(v.meanFakeScore)),
    row(`Top-${v.k ?? '?'} face average`, pct(v.topKFakeScore),
        'Mean of the strongest sustained face-level evidence'),
    row('Face-score variance', typeof v.scoreVariance === 'number' ? v.scoreVariance.toFixed(4) : '—')
  );
  if (v.originDetector && typeof v.originDetector.score === 'number') {
    rows.push(
      row('Full-frame AI-origin score', pct(v.originDetector.score),
          `Median of ${v.originDetector.frames ?? 'a few'} full-frame samples; uncalibrated supporting evidence`),
      row('Full-frame samples', v.originDetector.frames ?? '—')
    );
  }
  document.getElementById('video-stats').innerHTML = rows.join('');

  const timeline = Array.isArray(v.timeline) ? v.timeline : [];
  if (timeline.length) {
    const at = typeof v.suspiciousAt === 'number' ? v.suspiciousAt : 0.7;
    document.getElementById('video-timeline').innerHTML = timeline.map(f => {
      const hasScore = typeof f.p === 'number';
      const value = hasScore ? f.p : 0;
      const height = hasScore ? Math.max(2, Math.round(value * 100)) : 2;
      const label = hasScore ? `${Math.round(value * 100)}% fake` : 'no face evidence';
      return `<div class="vt-bar${hasScore && value >= at ? ' suspicious' : ''}"
                   style="height: ${height}%"
                   title="${DS.util.escapeHtml(String(f.t))}s — ${DS.util.escapeHtml(label)}"></div>`;
    }).join('');
    const last = timeline[timeline.length - 1];
    document.getElementById('vt-end').textContent = formatClock(last.t);
    document.getElementById('video-timeline-wrap').hidden = false;
  }

  const marks = Array.isArray(v.topTimestamps) ? v.topTimestamps : [];
  if (marks.length) {
    document.getElementById('video-marks').innerHTML = marks.map(m =>
      `<span class="vt-mark">${DS.util.escapeHtml(m.timestamp || formatClock(m.time))}
         <span class="vt-mark-score">${Math.round((m.score || 0) * 100)}%</span>
       </span>`).join('');
    document.getElementById('video-marks-wrap').hidden = false;
  }

  const t = v.temporal || {};
  const signals = [
    ['Frames with a face', t.facesFound != null && t.framesSampled != null
      ? `${t.facesFound} / ${t.framesSampled}` : null,
      'Frames where face-level temporal evidence was available'],
    ['Face position jitter', t.facePositionJitter,
      'How much the face moved around the frame, as a fraction of frame size'],
    ['Face size jitter', t.faceSizeJitter,
      'Relative spread of the face scale across the clip'],
    ['Landmark jitter', t.landmarkJitter,
      'Frame-to-frame movement of facial landmarks, in face widths'],
    ['Appearance continuity', t.appearanceContinuity,
      'Correlation between consecutive face crops; descriptive only'],
  ].filter(([, value]) => value !== null && value !== undefined);

  if (signals.length) {
    document.getElementById('video-temporal').innerHTML = signals.map(
      ([label, value, title]) => row(label, typeof value === 'number' ? value.toFixed(4) : value, title)
    ).join('');
    document.getElementById('video-temporal-wrap').hidden = false;
  }
  DS.icons();
}

function formatClock(seconds) {
  const total = Math.max(0, Math.round(Number(seconds) || 0));
  return `${String(Math.floor(total / 60)).padStart(2, '0')}:${String(total % 60).padStart(2, '0')}`;
}

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
    <span>Preview unavailable</span>`;
  wrap.appendChild(placeholder);
  DS.icons();
}

function renderMeta(scan, isVideo) {
  const fileEl = document.getElementById('meta-file');
  const name = scan.fileName || '—';
  fileEl.textContent = DS.util.truncate(name, 32);
  fileEl.title = name;
  document.getElementById('meta-type').textContent = isVideo ? 'Video' : 'Image';
  document.getElementById('meta-size').textContent = DS.util.formatBytes(scan.fileSize);
  document.getElementById('meta-source').textContent = scan.source === 'url' ? 'URL' : 'Direct upload';
  document.getElementById('meta-analyzed').textContent = DS.util.formatDate(scan.completedAt);
}

function renderModel() {
  const m = DS.api.MODEL;
  document.getElementById('model-list').innerHTML = `
    <div class="meta-row"><dt>Model</dt><dd class="mono">${DS.util.escapeHtml(m.name)}</dd></div>
    <div class="meta-row"><dt>Backend</dt><dd>${DS.util.escapeHtml(m.backend)}</dd></div>
    <div class="meta-row"><dt>Parameters</dt><dd class="mono">${DS.util.escapeHtml(m.params)}</dd></div>
    <div class="meta-row"><dt>Input size</dt><dd class="mono">${DS.util.escapeHtml(m.input)}</dd></div>
    <div class="meta-row"><dt>Device</dt><dd class="mono">${DS.util.escapeHtml(m.device)}</dd></div>`;
}

let verdictCtx = null;
function bandFor(scan, confidence) {
  const bands = DS.api.CERTAINTY || [];
  if (!bands.length) return null;
  return (scan.certainty && bands.find(b => b.key === scan.certainty))
      || bands.find(b => confidence >= b.from)
      || null;
}

function renderVerdictFacts(scan) {
  const facts = document.getElementById('verdict-facts');
  if (!facts) return;
  const model = DS.api.MODEL;
  const name = [model.model_name, model.version]
    .filter(v => v && v !== '—').join(' ') || model.name || '—';
  const rows = [
    ['Model', name || '—'],
    [scan.fileType === 'video' ? 'Frames analyzed' : 'Regions analyzed',
     scan.framesAnalyzed != null ? String(scan.framesAnalyzed) : '—'],
  ];
  facts.innerHTML = rows.map(([label, value]) => `
    <div class="meta-row"><dt>${DS.util.escapeHtml(label)}</dt><dd class="mono">${DS.util.escapeHtml(value)}</dd></div>`
  ).join('') + `
    <div class="meta-row"><dt>Engine</dt><dd><span class="badge" id="scan-engine-badge">—</span></dd></div>`;
  DS.server.paintBadge(document.getElementById('scan-engine-badge'),
                       scan.engine || (DS.api.MODE === 'simulated' ? 'simulated' : 'live'));
}

function renderWhy(scan) {
  const block = document.getElementById('verdict-why');
  const list = document.getElementById('why-list');
  if (!block || !list) return;
  const explain = scan.explain || {};
  const regions = Array.isArray(explain.regions) ? explain.regions : [];
  const items = regions.length ? regions
    : (explain.focusRegion ? [{ name: explain.focusRegion, weight: null }] : []);
  if (!items.length) return;
  list.innerHTML = items.map(r => {
    const label = String(r.name || '').replace(/^the\s+/i, '');
    const share = typeof r.weight === 'number'
      ? `<span class="why-bar"><span class="why-fill" style="width:${Math.round(r.weight * 100)}%"></span></span>`
      : '';
    return `<li><span class="why-name">${DS.util.escapeHtml(label)}</span>${share}</li>`;
  }).join('');
  block.hidden = false;
}

function renderCertainty() {
  if (!verdictCtx) return;
  const { scan, confidence, isFake, unit } = verdictCtx;
  const band = bandFor(scan, confidence);
  const chip = document.getElementById('certainty-chip');
  if (chip && band) {
    chip.textContent = band.label;
    chip.hidden = false;
  }
  const strength = band ? ` — ${band.label.toLowerCase()}` : '';
  document.getElementById('verdict-note').textContent = isFake
    ? `Detection confidence ${confidence}%${strength}. Evidence consistent with AI generation or media manipulation was found across the analyzed ${unit}.`
    : `Detection confidence ${confidence}%${strength}. The enabled detectors did not find strong synthetic/manipulation evidence across the analyzed ${unit}.`;
}

function renderVerdict(scan, isFake, isVideo) {
  const confidence = Math.max(0, Math.min(100, Math.round(scan.confidence || 0)));
  const ring = document.getElementById('ring-value');
  const C = 2 * Math.PI * 80;
  ring.classList.add(isFake ? 'fake' : 'real');
  ring.style.strokeDasharray = String(C);
  ring.style.strokeDashoffset = String(C);
  document.getElementById('conf-ring').setAttribute('aria-label', `Detection confidence ${confidence}%`);
  requestAnimationFrame(() => requestAnimationFrame(() => {
    ring.style.strokeDashoffset = String(C * (1 - confidence / 100));
  }));
  animatePercent(document.getElementById('conf-pct'), confidence);

  const badge = document.getElementById('verdict-badge');
  badge.classList.add(isFake ? 'badge-danger' : 'badge-success');
  badge.innerHTML = `
    <i data-lucide="${isFake ? 'alert-triangle' : 'shield-check'}" class="icon"></i>
    ${isFake ? 'Likely Synthetic / Manipulated' : 'Likely Real'}`;

  const risk = scan.riskLevel || (isFake ? 'High' : 'Low');
  const riskClass = { Low: 'badge-success', Medium: 'badge-warning', High: 'badge-danger' }[risk]
    || 'badge-warning';
  const chip = document.getElementById('risk-chip');
  chip.classList.add(riskClass);
  chip.textContent = `Risk level: ${risk}`;
  verdictCtx = { scan, confidence, isFake, unit: isVideo ? 'frames' : 'regions' };
  renderCertainty();
  DS.icons();
}

function bindFeedback(scan) {
  const ask = document.getElementById('feedback-ask');
  const thanks = document.getElementById('feedback-thanks');
  const card = document.getElementById('feedback-card');
  const key = DS.KEYS.FEEDBACK || 'ds_feedback';
  const prior = DS.store.get(key, []).find(f => f.scanId === scan.id);
  if (prior) {
    if (prior.skipped) card.hidden = true;
    else { ask.hidden = true; thanks.hidden = false; }
    return;
  }

  const remember = entry => {
    const local = DS.store.get(key, []);
    local.unshift({ ...entry, scanId: scan.id, at: new Date().toISOString() });
    DS.store.set(key, local.slice(0, 200));
  };
  const send = agree => {
    const record = {
      scanId: scan.id,
      prediction: scan.prediction,
      confidence: scan.confidence,
      fileType: scan.fileType,
      agree,
    };
    remember(record);
    fetch('/api/feedback', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(record),
    }).catch(() => {});
    ask.hidden = true;
    thanks.hidden = false;
    DS.toast('Thanks for the feedback', 'success', { duration: 2000 });
  };

  document.getElementById('fb-yes').addEventListener('click', () => send(true));
  document.getElementById('fb-no').addEventListener('click', () => send(false));
  document.getElementById('fb-skip').addEventListener('click', () => {
    remember({ skipped: true });
    card.hidden = true;
  });
}

function renderMetrics(scan) {
  document.getElementById('metric-time').textContent = DS.util.formatDuration(scan.processingTime);
  document.getElementById('metric-frames').textContent =
    scan.framesAnalyzed != null ? String(scan.framesAnalyzed) : '1';
  document.getElementById('metric-device').textContent = scan.device || 'CPU';
}

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
