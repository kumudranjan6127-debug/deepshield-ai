/* ============================================================
   DeepShield AI — api.js
   Analysis engine interface.

   Live inference is authoritative. The simulated engine exists only for
   explicit static/demo deployments and is always marked simulated.
   ============================================================ */

DS.api = {
  MODE: 'auto',
  ENDPOINT: '/api/analyze',

  async resolveMode() {
    if (DS.api.MODE !== 'auto') return DS.api.MODE;

    // Opening the static files directly is an explicit demo action: there is
    // no HTTP backend to be unavailable. The zero-dependency dev server also
    // advertises itself explicitly via /api/health with engine="echo".
    if (typeof location !== 'undefined' && location.protocol === 'file:') {
      return 'simulated';
    }

    const health = await DS.server.health();
    if (!health) return 'unavailable';
    if (health.engine === 'live') return 'live';
    if (health.engine === 'echo' || health.engine === 'simulated') return 'simulated';
    return 'unavailable';
  },

  MODEL: {
    name: 'MobileNetV3',
    version: '1.0.0',
    params: '—',
    input: '224 × 224',
    device: 'CPU',
    backend: '—',
  },

  CERTAINTY: [],

  async analyze(scan, hooks = {}) {
    const mode = await DS.api.resolveMode();
    if (mode === 'live') return DS.api._analyzeLive(scan, hooks);
    if (mode === 'simulated') return DS.api._analyzeSimulated(scan, hooks);

    const failure = new Error(
      'The analysis server is unavailable. No simulated verdict was generated.'
    );
    failure.code = 'SERVER_UNAVAILABLE';
    failure.status = 503;
    throw failure;
  },

  async _analyzeLive(scan, hooks = {}) {
    const { onStage = () => {}, onLog = () => {}, onProgress = () => {}, onEta = () => {} } = hooks;
    const isVideo = scan.fileType === 'video';
    const stages = DS.api._stages(isVideo);

    onLog('engine: DeepShield live (Flask · real inference)');
    onLog(`model:  ${DS.api.MODEL.name} (${DS.api.MODEL.params} params, CPU)`);
    onLog(`input:  ${scan.fileName}${scan.fileSize ? ` (${DS.util.formatBytes(scan.fileSize)})` : ''}`);

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
          scanId: scan.id || null,
          uploadId: scan.uploadId || null,
          url: scan.sourceUrl || null,
          fileName: scan.fileName,
          fileType: scan.fileType,
          fileSize: scan.fileSize,
          frameRate: DS.settings.get().frameRate,
        }),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        const failure = new Error(err.error || `Analysis failed (${res.status})`);
        failure.code = err.error_code || null;
        failure.status = res.status;
        throw failure;
      }
      const result = await res.json();
      result.engine = result.engine || 'live';
      onProgress(100);
      onStage(stages.length - 1, 'Complete');
      onLog(`done:   prediction=${result.prediction} uncalibrated_score=${result.confidence}%`);
      return result;
    } finally {
      clearInterval(tick);
    }
  },

  _stages(isVideo) {
    return isVideo
      ? [
          { label: 'Validating file', weight: 8 },
          { label: 'Extracting frames', weight: 22 },
          { label: 'Detecting face regions', weight: 18 },
          { label: 'Preprocessing frames', weight: 12 },
          { label: 'Running model inference', weight: 30 },
          { label: 'Aggregating results', weight: 10 },
        ]
      : [
          { label: 'Validating file', weight: 10 },
          { label: 'Detecting face regions', weight: 22 },
          { label: 'Preprocessing image', weight: 18 },
          { label: 'Running model inference', weight: 38 },
          { label: 'Computing model score', weight: 12 },
        ];
  },

  _verdictFor(scan) {
    const name = String(scan.fileName || '').toLowerCase();
    const seed = DS.util.hash(name + (scan.fileSize || 0));
    let prediction;
    if (name.includes('fake') || name.includes('synth') || name.includes('gen')) prediction = 'deepfake';
    else if (name.includes('real') || name.includes('orig')) prediction = 'real';
    else prediction = (seed % 100) < 42 ? 'deepfake' : 'real';
    const confidence = 72 + (seed % 26);
    const riskLevel = prediction === 'deepfake'
      ? (confidence >= 85 ? 'High' : 'Medium')
      : (confidence >= 80 ? 'Low' : 'Medium');
    const isVideo = scan.fileType === 'video';
    const framesAnalyzed = isVideo ? 24 + (seed % 37) : 1;
    return { prediction, confidence, riskLevel, framesAnalyzed, engine: 'simulated' };
  },

  _analyzeSimulated(scan, hooks) {
    const { onStage = () => {}, onLog = () => {}, onProgress = () => {}, onEta = () => {} } = hooks;
    const isVideo = scan.fileType === 'video';
    const stages = DS.api._stages(isVideo);
    const totalMs = isVideo ? 8200 : 5600;
    const started = performance.now();
    onLog('engine: Simulated (demo) — NO REAL DETECTION');
    onLog(`model:  ${DS.api.MODEL.name} (demo placeholder)`);
    onLog(`input:  ${scan.fileName} (${DS.util.formatBytes(scan.fileSize)})`);

    return new Promise(resolve => {
      let stageIdx = -1;
      const tick = setInterval(() => {
        const elapsed = performance.now() - started;
        const pct = Math.min(99, (elapsed / totalMs) * 100);
        onProgress(pct);
        onEta(Math.max(0, Math.ceil((totalMs - elapsed) / 1000)));
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
        }
        if (elapsed >= totalMs) {
          clearInterval(tick);
          const verdict = DS.api._verdictFor(scan);
          onProgress(100);
          onStage(stages.length - 1, 'Complete');
          onLog(`demo:   prediction=${verdict.prediction} uncalibrated_score=${verdict.confidence}%`);
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
