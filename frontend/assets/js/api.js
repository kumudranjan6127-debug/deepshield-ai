/* ============================================================
   DeepShield AI — api.js
   Analysis engine interface.

   V1 ships with a simulated engine so the frontend is fully
   demonstrable without a server. When the Flask backend lands,
   set DS.api.MODE = 'live' and implement the fetch branch —
   every page already talks only to DS.api.analyze().
   ============================================================ */

DS.api = {
  // 'auto'  — use the Flask engine when it answers, else the JS mock
  // 'live' / 'simulated' — force one (handy for testing)
  MODE: 'auto',
  ENDPOINT: '/api/analyze',     // Flask route (live mode)

  /* Resolves 'auto' once per page via the shared /api/health call, so a
     static deployment (GitHub Pages, file://) still demonstrates the whole
     flow — the UI already labels that state "Simulated (demo)". */
  async resolveMode() {
    if (DS.api.MODE !== 'auto') return DS.api.MODE;
    const health = await DS.server.health();
    return health && health.engine === 'live' ? 'live' : 'simulated';
  },

  /* Placeholders only. The running model reports itself through
     /api/health and DS.server.hydrate() overwrites these — nothing here
     names a specific variant, so a stale value can never be displayed. */
  MODEL: {
    name: 'MobileNetV3',
    version: '1.0.0',
    params: '—',
    input: '224 × 224',
    device: 'CPU',
    backend: '—',
  },

  /* Certainty bands, filled from /api/health. Empty until then, and
     deliberately so: the browser must never hold its own opinion about
     where "strong evidence" begins. */
  CERTAINTY: [],

  /**
   * Analyze a scan.
   * @param {object} scan  {id, fileName, fileType: 'image'|'video', fileSize, source}
   * @param {object} hooks {onStage(index, label), onLog(line), onProgress(pct), onEta(seconds)}
   * @returns {Promise<object>} result fields merged into the scan by the caller:
   *   {prediction:'real'|'deepfake', confidence, riskLevel, processingTime,
   *    framesAnalyzed, model, device, completedAt}
   */
  async analyze(scan, hooks = {}) {
    const mode = await DS.api.resolveMode();
    return mode === 'live'
      ? DS.api._analyzeLive(scan, hooks)
      : DS.api._analyzeSimulated(scan, hooks);
  },

  /* ---------- live (Flask + real MobileNetV3) ----------
     The server call is synchronous, so we drive the progress/stage/log
     hooks from a timer while waiting, then snap to 100% on the result. */
  async _analyzeLive(scan, hooks = {}) {
    const { onStage = () => {}, onLog = () => {}, onProgress = () => {}, onEta = () => {} } = hooks;
    const isVideo = scan.fileType === 'video';
    const stages = DS.api._stages(isVideo);

    onLog('engine: DeepShield live (Flask · real inference)');
    onLog(`model:  ${DS.api.MODEL.name} (${DS.api.MODEL.params} params, CPU)`);
    onLog(`input:  ${scan.fileName}${scan.fileSize ? ` (${DS.util.formatBytes(scan.fileSize)})` : ''}`);

    // Ease toward 92% while the server works (video takes longer on CPU)
    const est = isVideo ? 20000 : 5000;
    let pct = 0;
    let lastStage = -1;
    const tick = setInterval(() => {
      pct = Math.min(92, pct + (92 - pct) * 0.055);
      onProgress(pct);
      onEta(Math.max(1, Math.ceil((est * (1 - pct / 100)) / 1000)));

      let acc = 0, idx = 0;
      for (let i = 0; i < stages.length; i++) {
        acc += stages[i].weight;
        if (pct <= acc) { idx = i; break; }
        idx = i;
      }
      if (idx !== lastStage) {
        lastStage = idx;
        onStage(idx, stages[idx].label);
        onLog(`stage:  ${stages[idx].label.toLowerCase()} …`);
      }
    }, 300);

    try {
      const res = await fetch(DS.api.ENDPOINT, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          uploadId: scan.uploadId || null,   // staged file (from /api/upload)
          url: scan.sourceUrl || null,       // direct-MP4 scans
          fileName: scan.fileName,
          fileType: scan.fileType,
          fileSize: scan.fileSize,
          frameRate: DS.settings.get().frameRate,
        }),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.error || `Analysis failed (${res.status})`);
      }
      const result = await res.json();
      onProgress(100);
      onStage(stages.length - 1, 'Complete');
      onLog(`done:   prediction=${result.prediction} confidence=${result.confidence}%`);
      return result;
    } finally {
      clearInterval(tick);
    }
  },

  /* ---------- simulated ---------- */
  _stages(isVideo) {
    return isVideo
      ? [
          { label: 'Validating file',        weight: 8 },
          { label: 'Extracting frames',      weight: 22 },
          { label: 'Detecting face regions', weight: 18 },
          { label: 'Preprocessing frames',   weight: 12 },
          { label: 'Running model inference',weight: 30 },
          { label: 'Aggregating results',    weight: 10 },
        ]
      : [
          { label: 'Validating file',        weight: 10 },
          { label: 'Detecting face regions', weight: 22 },
          { label: 'Preprocessing image',    weight: 18 },
          { label: 'Running model inference',weight: 38 },
          { label: 'Computing confidence',   weight: 12 },
        ];
  },

  _verdictFor(scan) {
    const name = String(scan.fileName || '').toLowerCase();
    const seed = DS.util.hash(name + (scan.fileSize || 0));

    // Demo affordance: filenames steer the verdict, otherwise seeded.
    let prediction;
    if (name.includes('fake') || name.includes('synth') || name.includes('gen')) {
      prediction = 'deepfake';
    } else if (name.includes('real') || name.includes('orig')) {
      prediction = 'real';
    } else {
      prediction = (seed % 100) < 42 ? 'deepfake' : 'real';
    }

    const confidence = 72 + (seed % 26); // 72–97
    const riskLevel =
      prediction === 'deepfake'
        ? (confidence >= 85 ? 'High' : 'Medium')
        : (confidence >= 80 ? 'Low' : 'Medium');

    const isVideo = scan.fileType === 'video';
    const framesAnalyzed = isVideo ? 24 + (seed % 37) : 1;

    return { prediction, confidence, riskLevel, framesAnalyzed };
  },

  _analyzeSimulated(scan, hooks) {
    const { onStage = () => {}, onLog = () => {}, onProgress = () => {}, onEta = () => {} } = hooks;
    const isVideo = scan.fileType === 'video';
    const stages = DS.api._stages(isVideo);
    const totalMs = isVideo ? 8200 : 5600;
    const started = performance.now();

    const logLines = [
      `engine: DeepShield v1.0.0 (inference: CPU)`,
      `model:  ${DS.api.MODEL.name} loaded (${DS.api.MODEL.params} params)`,
      `input:  ${scan.fileName} (${DS.util.formatBytes(scan.fileSize)})`,
    ];
    logLines.forEach(onLog);

    return new Promise(resolve => {
      let stageIdx = -1;
      let pct = 0;

      const tick = setInterval(() => {
        const elapsed = performance.now() - started;
        pct = Math.min(99, (elapsed / totalMs) * 100);
        onProgress(pct);
        onEta(Math.max(0, Math.ceil((totalMs - elapsed) / 1000)));

        // Advance through weighted stages
        let acc = 0, idx = 0;
        for (let i = 0; i < stages.length; i++) {
          acc += stages[i].weight;
          if (pct <= acc) { idx = i; break; }
          idx = i;
        }
        if (idx !== stageIdx) {
          stageIdx = idx;
          onStage(idx, stages[idx].label);
          onLog(`stage:  ${stages[idx].label.toLowerCase()} …`);
          if (stages[idx].label === 'Running model inference') {
            onLog(`infer:  batch=1 size=224x224 threads=4`);
          }
        }

        if (elapsed >= totalMs) {
          clearInterval(tick);
          const verdict = DS.api._verdictFor(scan);
          onProgress(100);
          onStage(stages.length - 1, 'Complete');
          onLog(`done:   prediction=${verdict.prediction} confidence=${verdict.confidence}%`);
          resolve({
            ...verdict,
            processingTime: Math.round(performance.now() - started),
            model: DS.api.MODEL.name,
            device: DS.api.MODEL.device,
            completedAt: new Date().toISOString(),
          });
        }
      }, 120);
    });
  },
};
