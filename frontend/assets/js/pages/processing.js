/* ============================================================
   DeepShield AI — processing.js
   Live analysis view: scanner visual, progress readout,
   pipeline stage list, engine console. Drives DS.api.analyze
   hooks, then persists the merged scan and moves to results.
   ============================================================ */

(function () {
  'use strict';

  const MAX_LOG_LINES = 80;
  let started = false;   // double-run guard
  let cancelled = false; // set when the user confirms cancellation

  const el = id => document.getElementById(id);

  /* ---- Helpers ---- */

  /* Shorten very long file names, keeping the extension visible.
     Text is set via textContent so it is always escaped. */
  function truncateName(name, max = 42) {
    const s = String(name || 'file');
    if (s.length <= max) return s;
    return `${s.slice(0, max - 14)}…${s.slice(-12)}`;
  }

  /* ---- Stage list ---- */

  function renderStages(listEl, labels) {
    listEl.innerHTML = labels.map(label => `
      <li class="stage-row" data-state="pending">
        <span class="stage-status"><span class="stage-dot"></span></span>
        <span>${label}</span>
      </li>`).join('');
  }

  /* Set one row's visual state; no-op if unchanged (avoids flicker) */
  function setRowState(row, state) {
    if (row.dataset.state === state) return false;
    row.dataset.state = state;
    row.className = 'stage-row'
      + (state === 'done' ? ' done' : state === 'active' ? ' active' : '');
    row.querySelector('.stage-status').innerHTML =
      state === 'done'   ? '<i data-lucide="check" class="icon-sm"></i>'
      : state === 'active' ? '<span class="loader"></span>'
      :                      '<span class="stage-dot"></span>';
    return true;
  }

  /* Mark stage `index` active, everything before it done */
  function markStage(listEl, index) {
    let changed = false;
    DS.util.qsa('.stage-row', listEl).forEach((row, i) => {
      const state = i < index ? 'done' : i === index ? 'active' : 'pending';
      if (setRowState(row, state)) changed = true;
    });
    if (changed) DS.icons();
  }

  /* ---- Engine console ---- */

  function logLine(body, text) {
    const time = new Date().toTimeString().slice(0, 8);

    const line = document.createElement('div');
    line.className = 'console-line';

    const stamp = document.createElement('span');
    stamp.className = 'console-time';
    stamp.textContent = `[${time}] `;

    line.appendChild(stamp);
    line.appendChild(document.createTextNode(String(text)));
    body.appendChild(line);

    while (body.children.length > MAX_LOG_LINES) body.firstElementChild.remove();
    body.scrollTop = body.scrollHeight;
  }

  /* Shrink a preview to an archive-sized thumbnail (~10KB) so fifty of
     them still fit in localStorage. Resolves null on any failure. */
  function compactThumb(dataUrl, max = 220, quality = 0.6) {
    return new Promise(resolve => {
      if (!dataUrl) { resolve(null); return; }
      const img = new Image();
      img.onload = () => {
        try {
          const scale = Math.min(1, max / img.width);
          const canvas = document.createElement('canvas');
          canvas.width = Math.round(img.width * scale);
          canvas.height = Math.round(img.height * scale);
          canvas.getContext('2d').drawImage(img, 0, 0, canvas.width, canvas.height);
          resolve(canvas.toDataURL('image/jpeg', quality));
        } catch { resolve(null); }
      };
      img.onerror = () => resolve(null);
      img.src = dataUrl;
    });
  }

  /* ---- Completion ---- */

  function finish(merged, ui) {
    DS.session.set(DS.KEYS.SCAN, merged);

    // History: honour the auto-delete privacy setting.
    //  ON  → no media is kept, so older reports show no image (by design)
    //  OFF → keep a compact thumbnail so past reports stay complete;
    //        full-size previews would blow the ~5MB localStorage quota
    //        at 50 entries, and a silent quota failure loses the scan.
    const entry = { ...merged };
    if (DS.settings.get().autoDelete) {
      delete entry.previewDataUrl;
      delete entry.explain;
      DS.history.add(entry);
    } else {
      compactThumb(merged.previewDataUrl).then(small => {
        if (small) entry.previewDataUrl = small;
        else delete entry.previewDataUrl;
        DS.history.add(entry);
      });
    }

    // Success state: progress 100%, all stages done, ring → check
    ui.pct.textContent = '100%';
    ui.bar.style.width = '100%';
    ui.eta.textContent = 'Done';
    ui.title.textContent = 'Analysis complete';

    DS.util.qsa('.stage-row', ui.stageList).forEach(row => setRowState(row, 'done'));

    ui.scanner.classList.add('done');
    ui.core.innerHTML = '<i data-lucide="check-circle" class="icon-xl"></i>';
    ui.cancelBtn.disabled = true;
    DS.icons();

    setTimeout(() => window.location.replace('result.html'), 900);
  }

  /* ---- Boot ---- */

  document.addEventListener('DOMContentLoaded', () => {
    if (!DS.auth.guard()) return;

    const scan = DS.session.get(DS.KEYS.SCAN);
    if (!scan) { window.location.replace('dashboard.html'); return; }

    // Already analyzed (e.g. back-navigation) — don't re-run or re-log history
    if (scan.completedAt) { window.location.replace('result.html'); return; }

    if (started) return;
    started = true;

    const ui = {
      title:     el('proc-title'),
      sub:       el('proc-sub'),
      pct:       el('proc-pct'),
      eta:       el('proc-eta'),
      bar:       el('proc-bar'),
      stageList: el('stage-list'),
      console:   el('console-body'),
      scanner:   el('scanner'),
      core:      el('scanner-core'),
      cancelBtn: el('cancel-btn'),
    };

    const isVideo = scan.fileType === 'video';

    ui.title.textContent = `Analyzing ${truncateName(scan.fileName)}`;
    /* The model names itself; this line used to hardcode "MobileNetV3". */
    const describeRun = () => {
      const name = DS.api.MODEL.name && DS.api.MODEL.name !== '—'
        ? DS.api.MODEL.name : 'Model';
      ui.sub.textContent = `${name} inference · CPU · ${isVideo ? 'video' : 'image'}`;
    };
    describeRun();
    document.addEventListener('ds:server-ready', describeRun);

    // Stage labels come straight from the engine definition
    const labels = DS.api._stages(isVideo).map(s => s.label);
    renderStages(ui.stageList, labels);

    // Cancel → discard the in-flight scan; the flag stops finish() if the
    // engine resolves while the confirm modal is still open.
    el('cancel-confirm').addEventListener('click', () => {
      cancelled = true;
      DS.session.remove(DS.KEYS.SCAN);
      window.location.href = 'dashboard.html';
    });

    // Run the analysis
    DS.api.analyze(scan, {
      onStage: index => markStage(ui.stageList, Math.min(index, labels.length - 1)),
      onLog:   line  => logLine(ui.console, line),
      onProgress: pct => {
        ui.pct.textContent = `${Math.round(pct)}%`;
        ui.bar.style.width = `${pct}%`;
      },
      onEta: seconds => {
        ui.eta.textContent = `≈ ${seconds}s remaining`;
      },
    })
      .then(result => { if (!cancelled) finish({ ...scan, ...result }, ui); })
      .catch(error => { if (!cancelled) showFailure(error, scan); });
  });

  /* ---- Failure ----
     Every refusal the backend produces carries a stable `error_code` and a
     sentence written for a person. Both are shown. The hints below add the
     one thing the server cannot know: what this user should do next. */
  const RECOVERY = {
    UPLOAD_NOT_FOUND: 'Staged uploads are cleared after 30 minutes. Upload the file again.',
    TOO_LARGE:        'Try a smaller file, or trim the clip before uploading.',
    IMAGE_TOO_LARGE:  'Resize the image below 40 megapixels and try again.',
    IMAGE_TOO_SMALL:  'The image is too small to contain a usable face.',
    VIDEO_TOO_LONG:   'Trim the clip and upload the section you care about.',
    BAD_TYPE:         'Supported: JPG, PNG, WebP, MP4, MOV and WebM.',
    BAD_MAGIC:        'The file contents do not match its extension. It may be renamed or damaged.',
    BAD_MIME:         'The file contents do not match its extension. It may be renamed or damaged.',
    CORRUPT_MEDIA:    'The file could not be decoded. Try re-exporting it.',
    EMPTY_FILE:       'The file is empty.',
    BLOCKED_URL:      'That address points inside a private network, so it is not fetched.',
    INSECURE_URL:     'Only https:// links to a direct video file are accepted.',
    URL_NOT_VIDEO:    'That link is a web page, not a video file. Download the video and upload it.',
    RATE_LIMITED:     'Too many requests in a short time. Wait a minute and try again.',
    BUSY:             'The server is analysing something else. Try again in a moment.',
  };

  /* Worth another attempt without changing anything */
  const TRANSIENT = new Set(['RATE_LIMITED', 'BUSY', 'INTERNAL']);

  function showFailure(error, scan) {
    const panel = document.getElementById('proc-error');
    if (!panel) {
      DS.toast(error.message || 'Analysis failed.', 'error');
      return;
    }

    const code = error && error.code;
    document.getElementById('proc-error-message').textContent =
      (error && error.message) || 'The analysis could not be completed.';

    const hint = RECOVERY[code];
    const hintEl = document.getElementById('proc-error-hint');
    if (hint) {
      hintEl.textContent = hint;
      hintEl.hidden = false;
    }

    /* The code is what someone quotes when asking for help. */
    const codeEl = document.getElementById('proc-error-code');
    if (code) {
      codeEl.textContent = code;
      codeEl.hidden = false;
    }

    const retry = document.getElementById('proc-error-retry');
    if (TRANSIENT.has(code)) {
      /* A listener, not a `javascript:` href — the Content-Security-Policy
         blocks those, and a link that silently does nothing is worse than
         no link at all. */
      retry.innerHTML = '<i data-lucide="rotate-ccw" class="icon-sm"></i> Retry';
      retry.href = '#';
      retry.addEventListener('click', event => {
        event.preventDefault();
        window.location.reload();
      });
    } else {
      retry.href = scan && scan.fileType === 'video'
        ? 'upload-video.html' : 'upload-image.html';
    }

    /* Hide the machinery that is no longer running */
    ['stage-list', 'console-body'].forEach(id => {
      const el = document.getElementById(id);
      const card = el && el.closest('section');
      if (card) card.hidden = true;
    });
    panel.hidden = false;
    DS.icons();
  }
})();
