/* ============================================================
   DeepShield AI — api.js
   Analysis engine interface.

   V1 ships with a simulated engine so the frontend is fully
   demonstrable without a server. When the Flask backend lands,
   set DS.api.MODE = 'live' and implement the fetch branch —
   every page already talks only to DS.api.analyze().
   ============================================================ */

DS.api = {
  MODE: 'simulated',            // 'simulated' | 'live'
  ENDPOINT: '/api/analyze',     // Flask route (live mode)

  MODEL: {
    name: 'MobileNetV3-Small',
    version: '1.0.0',
    params: '2.5M',
    input: '224 × 224',
    device: 'CPU',
    backend: 'PyTorch',
  },

  /**
   * Analyze a scan.
   * @param {object} scan  {id, fileName, fileType: 'image'|'video', fileSize, source}
   * @param {object} hooks {onStage(index, label), onLog(line), onProgress(pct), onEta(seconds)}
   * @returns {Promise<object>} result fields merged into the scan by the caller:
   *   {prediction:'real'|'deepfake', confidence, riskLevel, processingTime,
   *    framesAnalyzed, model, device, completedAt}
   */
  analyze(scan, hooks = {}) {
    if (DS.api.MODE === 'live') return DS.api._analyzeLive(scan, hooks);
    return DS.api._analyzeSimulated(scan, hooks);
  },

  /* ---------- live (Flask) ---------- */
  async _analyzeLive(scan, hooks) {
    // Placeholder wiring for V2: POST multipart to Flask, poll or await JSON.
    const res = await fetch(DS.api.ENDPOINT, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(scan),
    });
    if (!res.ok) throw new Error(`Analysis failed (${res.status})`);
    return res.json();
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
