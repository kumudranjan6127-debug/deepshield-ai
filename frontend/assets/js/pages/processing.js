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

  /* ---- Completion ---- */

  function finish(merged, ui) {
    DS.session.set(DS.KEYS.SCAN, merged);

    // History: honor the auto-delete privacy setting — keep the media
    // thumbnail and heatmap out of durable storage (privacy + the ~5MB
    // localStorage quota). The session copy keeps everything for the
    // result/report pages.
    const entry = { ...merged };
    if (DS.settings.get().autoDelete) {
      delete entry.previewDataUrl;
      delete entry.explain;
    }
    DS.history.add(entry);

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
    ui.sub.textContent = `MobileNetV3 inference · CPU · ${isVideo ? 'video' : 'image'}`;

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
      .catch(() => {
        DS.toast('Analysis failed. Please try again.', 'error');
        setTimeout(() => window.location.replace('dashboard.html'), 1400);
      });
  });
})();
