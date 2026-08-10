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
  renderVideo(scan);
  renderVerdictFacts(scan);
  renderWhy(scan);
  bindFeedback(scan);

  // Model facts arrive from /api/health — repaint the card when they land
  document.addEventListener('ds:server-ready', renderModel);
  /* Bands arrive with health, which may land after the verdict is drawn. */
  document.addEventListener('ds:server-ready', renderCertainty);
  document.addEventListener('ds:server-ready', () => renderVerdictFacts(scan));
});

/* ---- Analysis insights: judge votes + sensitivity heatmap ---- */
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

/* ---- Video analysis ----
   Everything here is read from scan.video, which the backend fills for
   video scans only. An image scan, or a video analysed before Phase 6,
   simply leaves the card hidden — no placeholder numbers. */
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
  document.getElementById('video-stats').innerHTML = [
    row('Frames analyzed', v.framesAnalyzed ?? '—'),
    row('Suspicious frames', `${v.suspiciousFrames ?? '—'} / ${v.framesAnalyzed ?? '—'}`,
        `Frames scoring ${suspiciousAt} or higher on their own`),
    row('Peak fake score', pct(v.peakFakeScore)),
    row('Median fake score', pct(v.medianFakeScore)),
    row('Mean fake score', pct(v.meanFakeScore)),
    row(`Top-${v.k ?? '?'} average`, pct(v.topKFakeScore),
        'The strongest sustained evidence: the mean of the k highest-scoring frames'),
    row('Score variance', typeof v.scoreVariance === 'number'
        ? v.scoreVariance.toFixed(4) : '—',
        'How much the score moved from frame to frame'),
  ].join('');

  /* Timeline — one bar per sampled frame, coloured by whether that frame
     was suspicious on its own. */
  const timeline = Array.isArray(v.timeline) ? v.timeline : [];
  if (timeline.length) {
    const at = typeof v.suspiciousAt === 'number' ? v.suspiciousAt : 0.7;
    document.getElementById('video-timeline').innerHTML = timeline.map(f => {
      const height = Math.max(2, Math.round((f.p || 0) * 100));
      return `<div class="vt-bar${f.p >= at ? ' suspicious' : ''}"
                   style="height: ${height}%"
                   title="${DS.util.escapeHtml(String(f.t))}s — ${Math.round((f.p || 0) * 100)}% fake"></div>`;
    }).join('');
    const last = timeline[timeline.length - 1];
    document.getElementById('vt-end').textContent =
      DS.util.escapeHtml(formatClock(last.t));
    document.getElementById('video-timeline-wrap').hidden = false;
  }

  /* The timestamps worth scrubbing to */
  const marks = Array.isArray(v.topTimestamps) ? v.topTimestamps : [];
  if (marks.length) {
    document.getElementById('video-marks').innerHTML = marks.map(m =>
      `<span class="vt-mark">${DS.util.escapeHtml(m.timestamp || formatClock(m.time))}
         <span class="vt-mark-score">${Math.round((m.score || 0) * 100)}%</span>
       </span>`).join('');
    document.getElementById('video-marks-wrap').hidden = false;
  }

  /* Phase 6B consistency signals — shown, never counted */
  const t = v.temporal || {};
  const signals = [
    ['Frames with a face', t.facesFound != null && t.framesSampled != null
      ? `${t.facesFound} / ${t.framesSampled}` : null,
      'Frames where a face was detected. The rest were scored as whole frames.'],
    ['Face position jitter', t.facePositionJitter,
      'How much the face moved around the frame, as a fraction of frame size'],
    ['Face size jitter', t.faceSizeJitter,
      'Relative spread of the face\'s scale across the clip'],
    ['Landmark jitter', t.landmarkJitter,
      'Frame-to-frame movement of the five facial landmarks, in face widths'],
    ['Appearance continuity', t.appearanceContinuity,
      'Correlation between consecutive face crops: 1.00 is a perfectly smooth clip'],
  ].filter(([, value]) => value !== null && value !== undefined);

  if (signals.length) {
    document.getElementById('video-temporal').innerHTML = signals.map(
      ([label, value, title]) =>
        row(label, typeof value === 'number' ? value.toFixed(4) : value, title)
    ).join('');
    document.getElementById('video-temporal-wrap').hidden = false;
  }

  DS.icons();
}

function formatClock(seconds) {
  const total = Math.max(0, Math.round(Number(seconds) || 0));
  return `${String(Math.floor(total / 60)).padStart(2, '0')}:${String(total % 60).padStart(2, '0')}`;
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

/* ---- Certainty band ----
   The band table comes from /api/health, so no threshold is written down
   in the browser and the UI can never disagree with the backend about
   where "strong evidence" begins. A verdict analysed before Phase 5 has
   no `certainty` field, so its band is looked up from the confidence it
   was saved with — using the server's boundaries, not local ones. When
   there is no backend to ask, the chip simply stays hidden rather than
   guessing a label. */
let verdictCtx = null;

function bandFor(scan, confidence) {
  const bands = DS.api.CERTAINTY || [];
  if (!bands.length) return null;
  return (scan.certainty && bands.find(b => b.key === scan.certainty))
      || bands.find(b => confidence >= b.from)
      || null;
}

/* ---- What produced this verdict ----
   Model, frames and — most importantly — which engine ran. A simulated
   verdict is not a weaker verdict, it is a placeholder, and a result read
   back from history has no status page to consult. */
function renderVerdictFacts(scan) {
  const facts = document.getElementById('verdict-facts');
  if (!facts) return;

  /* "DeepShield V3-Max" rather than "MobileNetV3-Large" — the product and
     its training run are what identifies a verdict; the architecture is in
     the model card below. Both come from the model's own metadata. */
  const model = DS.api.MODEL;
  const name = [model.model_name, model.version]
    .filter(v => v && v !== '—').join(' ') || model.name || '—';
  const rows = [
    ['Model', name || '—'],
    [scan.fileType === 'video' ? 'Frames analyzed' : 'Regions analyzed',
     scan.framesAnalyzed != null ? String(scan.framesAnalyzed) : '—'],
  ];

  facts.innerHTML = rows.map(([label, value]) => `
    <div class="meta-row">
      <dt>${DS.util.escapeHtml(label)}</dt>
      <dd class="mono">${DS.util.escapeHtml(value)}</dd>
    </div>`).join('') + `
    <div class="meta-row">
      <dt>Engine</dt>
      <dd><span class="badge" id="scan-engine-badge">—</span></dd>
    </div>`;

  /* `scan.engine` is stamped on the verdict itself. Older scans predate it;
     rather than guess, fall back to what the server reports now and let the
     badge say "simulated" only when we actually know it was. */
  DS.server.paintBadge(document.getElementById('scan-engine-badge'),
                       scan.engine || (DS.api.MODE === 'simulated' ? 'simulated' : 'live'));
}

/* ---- Why? ----
   Regions the prediction actually leaned on, ranked by how far the score
   moved when each was hidden. Nothing here is generated prose. */
function renderWhy(scan) {
  const block = document.getElementById('verdict-why');
  const list = document.getElementById('why-list');
  if (!block || !list) return;

  const explain = scan.explain || {};
  const regions = Array.isArray(explain.regions) ? explain.regions : [];

  /* Older scans carry only the single focus region — still worth showing. */
  const items = regions.length
    ? regions
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

  /* "Detection confidence 94%" — never "94% probability it is fake". The
     model is uncalibrated, so the number ranks evidence; it does not
     estimate a frequency. */
  const strength = band ? ` — ${band.label.toLowerCase()}` : '';
  document.getElementById('verdict-note').textContent = isFake
    ? `Detection confidence ${confidence}%${strength}. Patterns consistent with synthetic manipulation were found across the analyzed ${unit}.`
    : `Detection confidence ${confidence}%${strength}. No manipulation artifacts were found across the analyzed ${unit}.`;
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
    .setAttribute('aria-label', `Detection confidence ${confidence}%`);

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

  /* Certainty band + one-sentence explanation. The wording is the
     server's, not ours — see renderCertainty. */
  verdictCtx = { scan, confidence, isFake, unit: isVideo ? 'frames' : 'regions' };
  renderCertainty();

  DS.icons();
}

/* ---- Verdict feedback ----
   An evaluation signal: how often the system is right in the wild.
   Sent to the backend (rating only, no media) and mirrored locally so
   the dashboard can show it without a server. Never a training label. */
function bindFeedback(scan) {
  const ask = document.getElementById('feedback-ask');
  const thanks = document.getElementById('feedback-thanks');

  const card = document.getElementById('feedback-card');

  /* Answered or skipped before → don't ask again */
  const prior = DS.store.get('ds_feedback', []).find(f => f.scanId === scan.id);
  if (prior) {
    if (prior.skipped) card.hidden = true;
    else { ask.hidden = true; thanks.hidden = false; }
    return;
  }

  /* Remember the outcome locally; `skipped` entries are excluded from stats */
  const remember = entry => {
    const local = DS.store.get('ds_feedback', []);
    local.unshift({ ...entry, scanId: scan.id, at: new Date().toISOString() });
    DS.store.set('ds_feedback', local.slice(0, 200));
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
    }).catch(() => { /* offline / frontend-only — the local copy stands */ });

    ask.hidden = true;
    thanks.hidden = false;
    DS.toast('Thanks for the feedback', 'success', { duration: 2000 });
  };

  document.getElementById('fb-yes').addEventListener('click', () => send(true));
  document.getElementById('fb-no').addEventListener('click', () => send(false));

  /* Skip: nothing is sent or scored — "I don't know" is a valid answer
     and guessing would pollute the accuracy signal. */
  document.getElementById('fb-skip').addEventListener('click', () => {
    remember({ skipped: true });
    card.hidden = true;
  });
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
