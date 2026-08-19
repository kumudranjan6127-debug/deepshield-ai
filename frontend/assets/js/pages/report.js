/* ============================================================
   DeepShield AI — report.js
   Resolves which scan to report on, populates the printable document,
   and drives the Download PDF (window.print) flow.
   ============================================================ */

document.addEventListener('DOMContentLoaded', () => {
  const user = DS.auth.guard();
  if (!user) return;

  document.getElementById('generated-at').textContent =
    DS.util.formatDate(new Date().toISOString());

  const scan = resolveScan();
  renderList(scan);
  if (!scan) { showEmpty(); return; }

  renderReport(scan);
  bindDownload(scan);
});

function reportLabel(scan) {
  if (scan.prediction !== 'deepfake') return 'Real';
  if (scan.findingType === 'ai_generated') return 'AI-generated';
  if (scan.findingType === 'face_manipulation') return 'Manipulated';
  return 'Synthetic / Manipulated';
}

function hasOriginEvidence(scan) {
  return Array.isArray(scan.ensemble)
    && scan.ensemble.some(v => v && v.kind === 'ai-origin/full-frame'
      && typeof v.pFake === 'number');
}

/* ---- All reports: one row per scan in history, newest first ---- */
function renderList(current) {
  const history = DS.history.all();
  if (!history.length) return;

  document.getElementById('report-count').textContent = history.length;
  document.getElementById('report-list').innerHTML = history.map(s => {
    const isFake = s.prediction === 'deepfake';
    const active = current && s.id === current.id ? ' active' : '';
    return `
      <a class="report-row${active}" href="report.html?id=${encodeURIComponent(s.id)}">
        <span class="report-row-main">
          <span class="report-row-name">${DS.util.escapeHtml(DS.util.truncate(s.fileName || '—', 34))}</span>
          <span class="report-row-meta text-xs">${DS.util.formatDate(s.completedAt)}</span>
        </span>
        <span class="badge ${isFake ? 'badge-danger' : 'badge-success'}">
          ${DS.util.escapeHtml(reportLabel(s))} · <span class="mono">${s.confidence}%</span>
        </span>
      </a>`;
  }).join('');

  document.getElementById('report-list-card').hidden = false;
  DS.icons();
}

function resolveScan() {
  const current = DS.session.get(DS.KEYS.SCAN);
  const id = new URLSearchParams(window.location.search).get('id');

  if (id) {
    const found = DS.history.find(id);
    if (found) {
      return (current && current.id === id) ? { ...current, ...found } : found;
    }
    if (current && current.id === id && current.prediction) return current;
  }

  if (current && current.prediction) return current;
  return DS.history.all()[0] || null;
}

function showEmpty() {
  document.getElementById('report-doc').hidden = true;
  document.getElementById('report-empty').hidden = false;
  document.getElementById('btn-download').disabled = true;
  DS.icons();
}

function renderReport(scan) {
  const set = (id, value) => { document.getElementById(id).textContent = value; };

  const settings = DS.settings.get();
  const model = DS.api.MODEL;
  const isFake = scan.prediction === 'deepfake';
  const isVideo = scan.fileType === 'video';
  const risk = scan.riskLevel || 'Medium';
  const label = reportLabel(scan);

  set('doc-id', scan.id || '—');
  set('doc-date', 'Generated ' + DS.util.formatDate(new Date().toISOString()));

  document.getElementById('verdict-strip').classList.add(isFake ? 'deepfake' : 'real');
  set('verdict-text', isFake ? `LIKELY ${label.toUpperCase()}` : 'LIKELY REAL');
  set('verdict-conf', `${scan.confidence}%`);

  const photo = scan.previewDataUrl || null;
  const heat = (scan.explain && scan.explain.heatmapDataUrl) || null;
  if (!photo && !heat && DS.settings.get().autoDelete) {
    document.getElementById('media-note').hidden = false;
  }

  if (photo || heat) {
    document.getElementById('media-section').hidden = false;
    if (photo) {
      document.getElementById('doc-photo').src = photo;
      document.getElementById('doc-photo-wrap').hidden = false;
    }
    if (heat) {
      document.getElementById('doc-heatmap').src = heat;
      document.getElementById('doc-heat-wrap').hidden = false;
    }
    if (scan.explain && scan.explain.note) {
      const focus = document.getElementById('doc-focus');
      focus.textContent = scan.explain.note;
      focus.hidden = false;
    }
  }

  const riskBadge = document.getElementById('risk-badge');
  const riskClass = { Low: 'badge-success', Medium: 'badge-warning', High: 'badge-danger' };
  riskBadge.className = `badge ${riskClass[risk] || 'badge-warning'}`;
  riskBadge.textContent = `${risk} risk`;

  set('f-name', scan.fileName || '—');
  set('f-type', isVideo ? 'Video' : 'Image');
  set('f-size', DS.util.formatBytes(scan.fileSize));
  set('f-source', scan.source === 'url' ? 'URL import' : 'Direct upload');
  set('f-uploaded', DS.util.formatDate(scan.createdAt));

  set('d-prediction', label);
  set('d-confidence', `${scan.confidence}%`);
  set('d-risk', risk);
  set('d-threshold', `${settings.threshold}%`);

  set('a-model', scan.model || model.name);
  set('a-backend', model.backend);
  set('a-params', model.params);
  set('a-input', model.input);
  set('a-device', scan.device || model.device);
  set('a-frames', scan.framesAnalyzed != null ? String(scan.framesAnalyzed) : (isVideo ? '—' : '1'));
  set('a-sampling', `${settings.frameRate} fps`);
  set('a-time', DS.util.formatDuration(scan.processingTime));

  set('summary-text', buildSummary(scan));
}

function scopeSentence(scan) {
  if (scan.faceFound === undefined) return '';
  if (scan.faceFound === false) {
    if (hasOriginEvidence(scan)) {
      return ' No reliable face crop was available, so the face-manipulation detector had no answer; '
        + 'a separate full-frame AI-origin detector still analysed the media. Its score is supporting, '
        + 'uncalibrated evidence and should be read independently of face-manipulation evidence.';
    }
    return ' No reliable face evidence was available, so a 50% face-model result should be read as '
      + 'inconclusive rather than evidence that the media is real.';
  }
  const faces = Number(scan.facesFound || 0);
  if (faces > 1) {
    return ` ${faces} faces were detected; each was analysed and the face-level verdict reports the `
      + 'most suspicious one, so one manipulated face can determine the result for the media.';
  }
  return '';
}

function buildSummary(scan) {
  const frames = scan.framesAnalyzed || 1;
  const frameTxt = `${frames} ${frames === 1 ? 'frame' : 'frames'}`;
  const media = scan.fileType === 'video' ? 'video' : 'image';
  const risk = (scan.riskLevel || 'Medium').toLowerCase();
  const origin = hasOriginEvidence(scan);

  if (scan.prediction === 'deepfake') {
    const kind = scan.findingType === 'ai_generated'
      ? 'AI-generation evidence'
      : scan.findingType === 'face_manipulation'
        ? 'face-manipulation evidence'
        : 'synthetic or manipulation evidence';
    return `The submitted ${media} was flagged with ${scan.confidence}% detection confidence, based on `
      + `analysis of ${frameTxt}. The enabled detectors found ${kind}, placing this media at ${risk} risk. `
      + `The confidence score is not a calibrated probability. Verify the original source before the `
      + `content is shared or relied upon.` + scopeSentence(scan);
  }

  const inconclusive = scan.insufficientEvidence === true
    || (scan.faceFound === false && !origin)
    || Number(scan.confidence) <= 50;
  if (inconclusive) {
    return `The submitted ${media} did not produce enough reliable evidence for a confident synthetic-media `
      + `decision. A 50% detection score represents an inconclusive result, not proof that the media is real. `
      + `Source verification or a second forensic method is recommended.` + scopeSentence(scan);
  }

  return `The submitted ${media} was classified as likely real with ${scan.confidence}% detection confidence, `
    + `based on analysis of ${frameTxt}. The enabled detectors did not find strong AI-generation or manipulation `
    + `evidence and the media is rated ${risk} risk. This score is not a calibrated probability; source `
    + `verification is still recommended for sensitive use cases.` + scopeSentence(scan);
}

function bindDownload(scan) {
  const btn = document.getElementById('btn-download');
  const originalTitle = document.title;
  window.addEventListener('afterprint', () => { document.title = originalTitle; });
  btn.addEventListener('click', () => {
    document.title = `DeepShield-Report-${scan.id || 'latest'}`;
    window.print();
  });
}
