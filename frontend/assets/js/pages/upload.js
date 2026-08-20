/* ============================================================
   DeepShield AI — pages/upload.js
   Shared controller for upload-image.html and upload-video.html.
   Branches on <body data-page="upload-image" | "upload-video">.

   Flow: pick/drop file → validate → preview + simulated upload
   progress → build scan object → hand off to processing.html.
   ============================================================ */

(function () {
  'use strict';

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

  const PREVIEW_MAX_W = 640;
  const PREVIEW_QUALITY = 0.7;
  const UPLOAD_SIM_MS = 900;

  // Backend deliberately rejects >40 MP images to avoid a decompression/RAM
  // spike on the small Render instance. Modern phones can produce 48/50/64 MP
  // JPEGs that are still well under the 10 MB file-size limit, so prepare a
  // safe analysis copy in the browser rather than rejecting a perfectly usable
  // photo. The detector caps images to 1024 px internally anyway.
  const MAX_UPLOAD_IMAGE_PIXELS = 36_000_000;
  const MAX_UPLOAD_IMAGE_SIDE = 4096;

  let cfg = null;
  let els = {};
  let currentFile = null;
  let objectUrl = null;
  let previewDataUrl = null;
  let progressTimer = null;
  let ready = false;

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
      img: q('#preview-img'),
      video: q('#preview-video'),
      metaName: q('#meta-name'),
      metaSize: q('#meta-size'),
      progressWrap: q('#progress-wrap'),
      progressPct: q('#progress-pct'),
      progressBar: q('#progress-bar'),
      readyLine: q('#preview-ready'),
      startBtn: q('#start-btn'),
      removeBtn: q('#remove-btn'),
      urlForm: q('#url-form'),
      urlField: q('#url-field'),
      urlInput: q('#url-input'),
      urlHint: q('#url-hint'),
      urlHintName: q('#url-hint-name'),
    };
  }

  function bindUpload() {
    const { area, input } = els;

    area.addEventListener('click', e => {
      if (e.target === input) return;
      input.click();
    });

    input.addEventListener('change', () => {
      if (input.files && input.files[0]) handleFile(input.files[0]);
    });

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

  function fileExt(name) {
    const parts = String(name || '').split('.');
    return parts.length > 1 ? parts.pop().toLowerCase() : '';
  }

  function validateFile(file) {
    const mimeOk = cfg.accept.includes(String(file.type || '').toLowerCase());
    const extOk = cfg.exts.includes(fileExt(file.name));
    if (!mimeOk && !extOk) return cfg.typeMsg;
    if (file.size > cfg.maxBytes) return cfg.sizeMsg;
    return null;
  }

  function showPreview(file) {
    if (objectUrl) URL.revokeObjectURL(objectUrl);
    objectUrl = URL.createObjectURL(file);
    previewDataUrl = null;

    if (cfg.kind === 'image' && els.img) {
      els.img.addEventListener('load', () => {
        previewDataUrl = makePreviewDataUrl(els.img, els.img.naturalWidth, els.img.naturalHeight);
      }, { once: true });
      els.img.src = objectUrl;
    } else if (els.video) {
      els.video.addEventListener('loadeddata', () => {
        previewDataUrl = makePreviewDataUrl(els.video, els.video.videoWidth, els.video.videoHeight);
        els.video.addEventListener('seeked', () => {
          const dataUrl = makePreviewDataUrl(els.video, els.video.videoWidth, els.video.videoHeight);
          if (dataUrl) previewDataUrl = dataUrl;
        }, { once: true });
        try { els.video.currentTime = 0.1; } catch { /* non-seekable */ }
      }, { once: true });
      els.video.src = objectUrl;
    }

    els.metaName.textContent = file.name;
    els.metaName.title = file.name;
    els.metaSize.textContent = DS.util.formatBytes(file.size);

    els.uploadCard.hidden = true;
    els.previewCard.hidden = false;
    runProgress();
  }

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

  function safeMultipartName(original, uploadFile) {
    const type = String(uploadFile.type || original.type || '').toLowerCase();
    const extByMime = {
      'image/jpeg': 'jpg',
      'image/png': 'png',
      'image/webp': 'webp',
      'video/mp4': 'mp4',
      'video/webm': 'webm',
      'video/quicktime': 'mov',
    };
    const wanted = extByMime[type];
    const ext = fileExt(original.name);
    if (!wanted || cfg.exts.includes(ext)) return original.name || `upload.${wanted || 'bin'}`;
    const base = String(original.name || 'upload').replace(/\.[^.]*$/, '') || 'upload';
    return `${base}.${wanted}`;
  }

  function prepareFileForUpload(file) {
    if (cfg.kind !== 'image' || !els.img || !els.img.complete) {
      return Promise.resolve({ file, optimized: false });
    }

    const width = els.img.naturalWidth;
    const height = els.img.naturalHeight;
    if (!width || !height) return Promise.resolve({ file, optimized: false });

    const pixels = width * height;
    if (pixels <= MAX_UPLOAD_IMAGE_PIXELS && Math.max(width, height) <= MAX_UPLOAD_IMAGE_SIDE) {
      return Promise.resolve({ file, optimized: false });
    }

    const scale = Math.min(
      1,
      MAX_UPLOAD_IMAGE_SIDE / Math.max(width, height),
      Math.sqrt(MAX_UPLOAD_IMAGE_PIXELS / pixels),
    );
    const outW = Math.max(1, Math.floor(width * scale));
    const outH = Math.max(1, Math.floor(height * scale));

    return new Promise(resolve => {
      try {
        const canvas = document.createElement('canvas');
        canvas.width = outW;
        canvas.height = outH;
        const ctx = canvas.getContext('2d');
        if (!ctx) { resolve({ file, optimized: false }); return; }
        ctx.drawImage(els.img, 0, 0, outW, outH);
        canvas.toBlob(blob => {
          if (!blob) { resolve({ file, optimized: false }); return; }
          const base = String(file.name || 'photo').replace(/\.[^.]*$/, '') || 'photo';
          const optimized = new File([blob], `${base}.jpg`, {
            type: 'image/jpeg',
            lastModified: file.lastModified || Date.now(),
          });
          resolve({ file: optimized, optimized: true });
        }, 'image/jpeg', 0.90);
      } catch {
        resolve({ file, optimized: false });
      }
    });
  }

  async function startAnalysis() {
    if (!currentFile || !ready) return;

    const mode = await DS.api.resolveMode();
    if (mode === 'unavailable') {
      DS.toast('The analysis server is unavailable. No simulated verdict was generated.', 'error');
      return;
    }

    const scan = {
      id: DS.util.uid(),
      fileName: currentFile.name,
      fileType: cfg.kind,
      fileSize: currentFile.size,
      source: 'upload',
      previewDataUrl: previewDataUrl,
      createdAt: new Date().toISOString(),
    };

    if (mode === 'live') {
      const originalLabel = els.startBtn.innerHTML;
      els.startBtn.disabled = true;
      els.removeBtn.disabled = true;
      els.startBtn.innerHTML = '<span class="loader" aria-hidden="true"></span> Preparing…';

      try {
        const prepared = await prepareFileForUpload(currentFile);
        const uploadFile = prepared.file;
        if (prepared.optimized) {
          els.startBtn.innerHTML = '<span class="loader" aria-hidden="true"></span> Optimizing photo…';
        }

        const fd = new FormData();
        fd.append('file', uploadFile, safeMultipartName(currentFile, uploadFile));
        els.startBtn.innerHTML = '<span class="loader" aria-hidden="true"></span> Uploading…';

        const res = await fetch('/api/upload', { method: 'POST', body: fd });
        const payload = await res.json().catch(() => ({}));
        if (!res.ok) {
          const failure = new Error(payload.error || `Upload failed (${res.status})`);
          failure.code = payload.error_code || null;
          failure.status = res.status;
          throw failure;
        }
        if (!payload.uploadId) {
          const failure = new Error('The server accepted the upload but did not return an upload ID.');
          failure.code = 'BAD_UPLOAD_RESPONSE';
          throw failure;
        }
        scan.uploadId = payload.uploadId;
        scan.uploadOptimized = prepared.optimized;
      } catch (error) {
        const code = error && error.code;
        let message = (error && error.message) || 'Could not reach the analysis server.';
        if (!code && (!error || error.name === 'TypeError')) {
          message = 'Could not reach the analysis server. Check your connection and try again.';
        } else if (code === 'IMAGE_TOO_LARGE') {
          message = 'This photo is too large for the server. Try a normal camera photo instead of 48/50/64 MP mode.';
        } else if (code === 'RATE_LIMITED') {
          message = error.message || 'Too many attempts. Wait a moment and try again.';
        }
        DS.toast(message, 'error', code ? { title: code } : {});
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

  function bindUrlForm() {
    if (!els.urlForm) return;

    els.urlInput.addEventListener('input', () =>
      els.urlField.classList.remove('invalid'));

    els.urlForm.addEventListener('submit', e => {
      e.preventDefault();
      const raw = els.urlInput.value.trim();
      const platform = streamingPlatform(raw);
      if (platform) {
        showUrlHint(platform);
        return;
      }

      const parsed = parseVideoUrl(raw);
      if (!parsed) {
        els.urlField.classList.add('invalid');
        hideUrlHint();
        DS.toast('Enter a direct http(s) link ending in .mp4', 'error');
        return;
      }
      hideUrlHint();
      const scan = {
        id: DS.util.uid(),
        fileName: parsed.fileName,
        fileType: 'video',
        fileSize: null,
        source: 'url',
        sourceUrl: els.urlInput.value.trim(),
        previewDataUrl: null,
        createdAt: new Date().toISOString(),
      };
      DS.session.set(DS.KEYS.SCAN, scan);
      window.location.href = 'processing.html';
    });
  }

  const PLATFORMS = [
    { name: 'YouTube', hosts: ['youtube.com', 'youtu.be', 'youtube-nocookie.com'] },
    { name: 'Instagram', hosts: ['instagram.com', 'instagr.am'] },
    { name: 'Facebook', hosts: ['facebook.com', 'fb.watch'] },
    { name: 'TikTok', hosts: ['tiktok.com'] },
    { name: 'X (Twitter)', hosts: ['twitter.com', 'x.com'] },
    { name: 'Reddit', hosts: ['reddit.com', 'redd.it'] },
    { name: 'Vimeo', hosts: ['vimeo.com'] },
    { name: 'Dailymotion', hosts: ['dailymotion.com', 'dai.ly'] },
    { name: 'Snapchat', hosts: ['snapchat.com'] },
    { name: 'Telegram', hosts: ['t.me', 'telegram.me'] },
  ];

  function streamingPlatform(raw) {
    if (!raw) return null;
    let host;
    try {
      host = new URL(/^https?:\/\//i.test(raw) ? raw : 'https://' + raw)
        .hostname.toLowerCase().replace(/^www\./, '');
    } catch { return null; }
    const hit = PLATFORMS.find(p => p.hosts.some(h => host === h || host.endsWith('.' + h)));
    return hit ? hit.name : null;
  }

  function showUrlHint(platform) {
    els.urlField.classList.remove('invalid');
    els.urlHintName.textContent = platform;
    els.urlHint.hidden = false;
    DS.icons();
  }

  function hideUrlHint() {
    if (els.urlHint) els.urlHint.hidden = true;
  }

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