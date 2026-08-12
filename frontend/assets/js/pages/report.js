/* ============================================================
   DeepShield AI — report.js
   Resolves which scan to report on, populates the printable
   document, and drives the Download PDF (window.print) flow.
   ============================================================ */

document.addEventListener('DOMContentLoaded', () => {
  const user = DS.auth.guard();
  if (!user) return;

  /* Page-head "Generated" timestamp */
  document.getElementById('generated-at').textContent =
    DS.util.formatDate(new Date().toISOString());

  const scan = resolveScan();
  renderList(scan);              // every past report stays reachable
  if (!scan) { showEmpty(); return; }

  renderReport(scan);
  bindDownload(scan);
});

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
          ${isFake ? 'Deepfake' : 'Real'} · <span class="mono">${s.confidence}%</span>
        </span>
      </a>`;
  }).join('');

  document.getElementById('report-list-card').hidden = false;
  DS.icons();
}

/* ---- Scan resolution: ?id → in-flight session scan → latest history ----
   History entries drop previewDataUrl/explain when auto-delete is on, so
   when the requested scan is also the one in this session, merge the two:
   history for the record, session for the media. */
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

/* ---- Empty state (no completed scans anywhere) ---- */
function showEmpty() {
  document.getElementById('report-doc').hidden = true;
  document.getElementById('report-empty').hidden = false;
  document.getElementById('btn-download').disabled = true;
  DS.icons();
}

/* ---- Populate the document ---- */
function renderReport(scan) {
  const set = (id, value) => { document.getElementById(id).textContent = value; };

  const settings = DS.settings.get();
  const model = DS.api.MODEL;
  const isFake = scan.prediction === 'deepfake';
  const isVideo = scan.fileType === 'video';
  const risk = scan.riskLevel || 'Medium';

  /* 1. Doc header */
  set('doc-id', scan.id || '—');
  set('doc-date', 'Generated ' + DS.util.formatDate(new Date().toISOString()));

  /* 2. Verdict strip */
  document.getElementById('verdict-strip').classList.add(isFake ? 'deepfake' : 'real');
  set('verdict-text', isFake ? 'LIKELY DEEPFAKE' : 'LIKELY REAL');
  set('verdict-conf', `${scan.confidence}%`);

  /* 2b. Analyzed media — photo + sensitivity heatmap (when available) */
  const photo = scan.previewDataUrl || null;
  const heat = (scan.explain && scan.explain.heatmapDataUrl) || null;

  // No media: say why rather than silently dropping the section
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

  /* 3. File details */
  set('f-name', scan.fileName || '—');
  set('f-type', isVideo ? 'Video' : 'Image');
  set('f-size', DS.util.formatBytes(scan.fileSize));
  set('f-source', scan.source === 'url' ? 'URL import' : 'Direct upload');
  set('f-uploaded', DS.util.formatDate(scan.createdAt));

  /* 4. Detection outcome */
  set('d-prediction', isFake ? 'Deepfake' : 'Real');
  set('d-confidence', `${scan.confidence}%`);
  set('d-risk', risk);
  set('d-threshold', `${settings.threshold}%`);

  /* 5. Analysis details */
  set('a-model', scan.model || model.name);
  set('a-backend', model.backend);
  set('a-params', model.params);
  set('a-input', model.input);
  set('a-device', scan.device || model.device);
  set('a-frames', scan.framesAnalyzed != null ? String(scan.framesAnalyzed) : (isVideo ? '—' : '1'));
  set('a-sampling', `${settings.frameRate} fps`);
  set('a-time', DS.util.formatDuration(scan.processingTime));

  /* 6. Summary */
  set('summary-text', buildSummary(scan));
}

/* ---- What the verdict is about ----
   A report is the artefact someone forwards to a third party, so anything
   qualifying the verdict has to travel with it. "No significant
   manipulation artifacts were detected" is simply false about a picture
   with no face in it, and incomplete about a group photograph where one
   face out of four decided the answer.

   Empty for scans that predate the flag, so old reports read as before. */
function scopeSentence(scan) {
  if (scan.faceFound === undefined) return '';
  if (scan.faceFound === false) {
    return ' No face was detected, so the whole frame was analysed. This '
      + 'model is trained on faces, and a score produced this way is not '
      + 'evidence about the media - it should be read as no result.';
  }
  const faces = Number(scan.facesFound || 0);
  if (faces > 1) {
    return ` ${faces} faces were detected; each was analysed and the verdict `
      + 'reports the most suspicious of them, so one manipulated face '
      + 'determines the result for the whole image.';
  }
  return '';
}


/* ---- Summary paragraph, templated per verdict ---- */
function buildSummary(scan) {
  const frames = scan.framesAnalyzed || 1;
  const frameTxt = `${frames} ${frames === 1 ? 'frame' : 'frames'}`;
  const media = scan.fileType === 'video' ? 'video' : 'image';
  const risk = (scan.riskLevel || 'Medium').toLowerCase();

  if (scan.prediction === 'deepfake') {
    return `The submitted ${media} was classified as a likely deepfake with ${scan.confidence}% model `
      + `confidence, based on analysis of ${frameTxt}. Detected patterns are consistent with `
      + `synthetically generated or manipulated facial content, placing this media at ${risk} risk. `
      + `We recommend verifying the original source before this content is shared or relied upon.`
      + scopeSentence(scan);
  }
  return `The submitted ${media} was classified as likely authentic with ${scan.confidence}% model `
    + `confidence, based on analysis of ${frameTxt}. No significant manipulation artifacts were `
    + `detected, and the media is rated ${risk} risk. As with any automated screening, pairing this `
    + `result with source verification is recommended for sensitive use cases.`
    + scopeSentence(scan);
}

/* ---- Download PDF = retitle document, open the print dialog ---- */
function bindDownload(scan) {
  const btn = document.getElementById('btn-download');
  const originalTitle = document.title;

  window.addEventListener('afterprint', () => { document.title = originalTitle; });

  btn.addEventListener('click', () => {
    document.title = `DeepShield-Report-${scan.id || 'latest'}`;
    window.print();
  });
}
