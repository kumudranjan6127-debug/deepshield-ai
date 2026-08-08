/* ============================================================
   DeepShield AI — pages/upload.js
   Shared controller for upload-image.html and upload-video.html.
   Branches on <body data-page="upload-image" | "upload-video">.

   Flow: pick/drop file → validate → preview + simulated upload
   progress → build scan object → hand off to processing.html.
   ============================================================ */

(function () {
  'use strict';

  /* ============ PAGE-TYPE CONFIG ============ */

  const CONFIGS = {
    'upload-image': {
      kind: 'image',
      accept: ['image/jpeg', 'image/png', 'image/webp'],
      exts: ['jpg', 'jpeg', 'png', 'webp'],
      maxBytes: 10 * 1024 * 1024,
      typeMsg: 'Unsupported file type — use JPG, PNG or WebP.',
      sizeMsg: 'Image is too large — the limit is 10 MB.',
    },
    'upload-video': {
      kind: 'video',
      accept: ['video/mp4', 'video/webm', 'video/quicktime'],
      exts: ['mp4', 'webm', 'mov'],
      maxBytes: 100 * 1024 * 1024,
      typeMsg: 'Unsupported file type — use MP4, WebM or MOV.',
      sizeMsg: 'Video is too large — the limit is 100 MB.',
    },
  };

  const PREVIEW_MAX_W = 640;   // px — thumbnail width cap for previewDataUrl
  const PREVIEW_QUALITY = 0.7; // JPEG quality for previewDataUrl
  const UPLOAD_SIM_MS = 900;   // simulated upload duration

  /* ============ STATE ============ */

  let cfg = null;
  let els = {};
  let currentFile = null;
  let objectUrl = null;       // blob URL backing the visible preview
  let previewDataUrl = null;  // downscaled JPEG data URL (or null)
  let progressTimer = null;
  let ready = false;          // simulated upload finished

  /* ============ BOOT ============ */

  document.addEventListener('DOMContentLoaded', () => {
    if (!DS.auth.guard()) return;

    cfg = CONFIGS[document.body.dataset.page];
    if (!cfg) return;

    cacheEls();
    bindUpload();
    if (cfg.kind === 'video') bindUrlForm();
  });

  function cacheEls() {
    const q = DS.util.qs;
    els = {
      uploadCard: q('#upload-card'),
      area: q('#upload-area'),
      input: q('#file-input'),
      previewCard: q('#preview-card'),
      img: q('#preview-img'),          // image page only
      video: q('#preview-video'),      // video page only
      metaName: q('#meta-name'),
      metaSize: q('#meta-size'),
      progressWrap: q('#progress-wrap'),
      progressPct: q('#progress-pct'),
      progressBar: q('#progress-bar'),
      readyLine: q('#preview-ready'),
      startBtn: q('#start-btn'),
      removeBtn: q('#remove-btn'),
      urlForm: q('#url-form'),         // video page only
      urlField: q('#url-field'),
      urlInput: q('#url-input'),
    };
  }

  /* ============ DROP ZONE / FILE PICKING ============ */

  function bindUpload() {
    const { area, input } = els;

    // Whole area opens the picker (the Browse button's click bubbles here)
    area.addEventListener('click', e => {
      if (e.target === input) return; // ignore the programmatic click echo
      input.click();
    });


    input.addEventListener('change', () => {
      if (input.files && input.files[0]) handleFile(input.files[0]);
    });

    // Drag & drop
    ['dragenter', 'dragover'].forEach(type =>
      area.addEventListener(type, e => {
        e.preventDefault();
        area.classList.add('dragover');
      }));
    ['dragleave', 'dragend'].forEach(type =>
      area.addEventListener(type, () => area.classList.remove('dragover')));
    area.addEventListener('drop', e => {
      e.preventDefault();
      area.classList.remove('dragover');
      const file = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
      if (file) handleFile(file);
    });

    // Preview actions
    els.startBtn.addEventListener('click', startAnalysis);
    els.removeBtn.addEventListener('click', resetState);
  }

  function handleFile(file) {
    const error = validateFile(file);
    if (error) {
      DS.toast(error, 'error');
      els.input.value = '';
      return;
    }
    currentFile = file;
    showPreview(file);
  }

  /* ============ VALIDATION ============ */

  function fileExt(name) {
    const parts = String(name || '').split('.');
    return parts.length > 1 ? parts.pop().toLowerCase() : '';
  }

  /** Returns an error message, or null when the file is acceptable. */
  function validateFile(file) {
    const mimeOk = cfg.accept.includes(String(file.type || '').toLowerCase());
    const extOk = cfg.exts.includes(fileExt(file.name));
    if (!mimeOk && !extOk) return cfg.typeMsg;
    if (file.size > cfg.maxBytes) return cfg.sizeMsg;
    return null;
  }

  /* ============ PREVIEW ============ */

  function showPreview(file) {
    if (objectUrl) URL.revokeObjectURL(objectUrl);
    objectUrl = URL.createObjectURL(file);
    previewDataUrl = null;

    // Media element + thumbnail capture (listeners first, then src)
    if (cfg.kind === 'image' && els.img) {
      els.img.addEventListener('load', () => {
        previewDataUrl = makePreviewDataUrl(els.img, els.img.naturalWidth, els.img.naturalHeight);
      }, { once: true });
      els.img.src = objectUrl;
    } else if (els.video) {
      els.video.addEventListener('loadeddata', () => {
        // Fallback capture at frame 0, then a cleaner grab just past it
        previewDataUrl = makePreviewDataUrl(els.video, els.video.videoWidth, els.video.videoHeight);
        els.video.addEventListener('seeked', () => {
          const dataUrl = makePreviewDataUrl(els.video, els.video.videoWidth, els.video.videoHeight);
          if (dataUrl) previewDataUrl = dataUrl;
        }, { once: true });
        try { els.video.currentTime = 0.1; } catch { /* non-seekable — keep frame 0 */ }
      }, { once: true });
      els.video.src = objectUrl;
    }

    // Meta row
    els.metaName.textContent = file.name;
    els.metaName.title = file.name;
    els.metaSize.textContent = DS.util.formatBytes(file.size);

    // Swap cards
    els.uploadCard.hidden = true;
    els.previewCard.hidden = false;

    runProgress();
  }

  /**
   * Draw a media element to a canvas, downscaled to PREVIEW_MAX_W,
   * and return a JPEG data URL. Returns null when the canvas is
   * tainted (file:// edge cases) or dimensions are unavailable.
   */
  function makePreviewDataUrl(source, width, height) {
    try {
      if (!width || !height) return null;
      const scale = Math.min(1, PREVIEW_MAX_W / width);
      const canvas = document.createElement('canvas');
      canvas.width = Math.round(width * scale);
      canvas.height = Math.round(height * scale);
      canvas.getContext('2d').drawImage(source, 0, 0, canvas.width, canvas.height);
      return canvas.toDataURL('image/jpeg', PREVIEW_QUALITY);
    } catch {
      return null;
    }
  }

  /* ============ SIMULATED UPLOAD PROGRESS ============ */

  function runProgress() {
    clearInterval(progressTimer);
    ready = false;
    els.startBtn.disabled = true;
    els.readyLine.hidden = true;
    els.progressWrap.hidden = false;
    els.progressBar.style.width = '0%';
    els.progressPct.textContent = '0%';

    const started = performance.now();
    progressTimer = setInterval(() => {
      const pct = Math.min(100, Math.round(((performance.now() - started) / UPLOAD_SIM_MS) * 100));
      els.progressBar.style.width = pct + '%';
      els.progressPct.textContent = pct + '%';
      if (pct >= 100) {
        clearInterval(progressTimer);
        progressTimer = null;
        ready = true;
        els.progressWrap.hidden = true;
        els.readyLine.hidden = false;
        els.startBtn.disabled = false;
      }
    }, 40);
  }

  /* ============ RESET ============ */

  function resetState() {
    clearInterval(progressTimer);
    progressTimer = null;
    ready = false;
    currentFile = null;
    previewDataUrl = null;

    if (objectUrl) {
      URL.revokeObjectURL(objectUrl);
      objectUrl = null;
    }
    if (els.img) els.img.removeAttribute('src');
    if (els.video) {
      els.video.pause();
      els.video.removeAttribute('src');
      els.video.load();
    }

    els.input.value = '';
    els.progressBar.style.width = '0%';
    els.progressPct.textContent = '0%';
    els.readyLine.hidden = true;
    els.progressWrap.hidden = false;
    els.startBtn.disabled = true;

    els.previewCard.hidden = true;
    els.uploadCard.hidden = false;
  }

  /* ============ HANDOFF → PROCESSING ============ */

  async function startAnalysis() {
    if (!currentFile || !ready) return;
    const scan = {
      id: DS.util.uid(),
      fileName: currentFile.name,
      fileType: cfg.kind,
      fileSize: currentFile.size,
      source: 'upload',
      previewDataUrl: previewDataUrl,
      createdAt: new Date().toISOString(),
    };

    // Live engine: a File object can't survive page navigation, so stage
    // it on the server now; processing.html analyzes it via uploadId.
    // Without a backend we skip this and the mock engine takes over.
    if (await DS.api.resolveMode() === 'live') {
      const originalLabel = els.startBtn.innerHTML;
      els.startBtn.disabled = true;
      els.removeBtn.disabled = true;
      els.startBtn.innerHTML = '<span class="loader" aria-hidden="true"></span> Uploading…';
      try {
        const fd = new FormData();
        fd.append('file', currentFile);
        const res = await fetch('/api/upload', { method: 'POST', body: fd });
        if (!res.ok) throw new Error();
        scan.uploadId = (await res.json()).uploadId;
      } catch {
        DS.toast('Could not reach the analysis server — is the backend running?', 'error');
        els.startBtn.disabled = false;
        els.removeBtn.disabled = false;
        els.startBtn.innerHTML = originalLabel;
        DS.icons();
        return;
      }
    }

    DS.session.set(DS.KEYS.SCAN, scan);
    window.location.href = 'processing.html';
  }

  /* ============ ANALYZE FROM URL (video page only) ============ */

  function bindUrlForm() {
    if (!els.urlForm) return;

    els.urlInput.addEventListener('input', () =>
      els.urlField.classList.remove('invalid'));

    els.urlForm.addEventListener('submit', e => {
      e.preventDefault();
      const parsed = parseVideoUrl(els.urlInput.value.trim());
      if (!parsed) {
        els.urlField.classList.add('invalid');
        DS.toast('Enter a direct http(s) link ending in .mp4', 'error');
        return;
      }
      const scan = {
        id: DS.util.uid(),
        fileName: parsed.fileName,
        fileType: 'video',
        fileSize: null,
        source: 'url',
        sourceUrl: els.urlInput.value.trim(), // live engine downloads from here
        previewDataUrl: null,
        createdAt: new Date().toISOString(),
      };
      DS.session.set(DS.KEYS.SCAN, scan);
      window.location.href = 'processing.html';
    });
  }

  /** Accepts http(s) URLs whose path ends in .mp4 (query strings OK). */
  function parseVideoUrl(raw) {
    if (!raw) return null;
    let url;
    try { url = new URL(raw); } catch { return null; }
    if (url.protocol !== 'http:' && url.protocol !== 'https:') return null;
    if (!url.pathname.toLowerCase().endsWith('.mp4')) return null;

    const segment = url.pathname.split('/').filter(Boolean).pop() || 'video.mp4';
    let fileName;
    try { fileName = decodeURIComponent(segment); } catch { fileName = segment; }
    return { fileName };
  }

})();
