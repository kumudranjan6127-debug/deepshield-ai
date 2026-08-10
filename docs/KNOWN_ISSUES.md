# Known Issues

Every entry below has been **observed**, not suspected. Where a number is
given, it was measured. Last reviewed after Phase 3, 2026-08-10.

Severity: 🔴 wrong output or wrong information · 🟡 works but flawed ·
🔵 limitation to document rather than fix

---

## 🟡 1. `ds_feedback` is not registered in `DS.KEYS`

`DS.KEYS` lists `ds_user`, `ds_scan`, `ds_history`, `ds_settings`. The
feedback store is written as a raw string in three places
(`pages/result.js` ×2, `pages/dashboard.js` ×1). It works, but it sits outside
the convention every other key follows, and "Clear history" does not clear it.

**Fix:** add `FEEDBACK: 'ds_feedback'` to `DS.KEYS` and use it.

## 🟡 2. OpenCV's DNN engine crashes on large batches

Discovered while building the occlusion heatmap: a 36-image batch killed the
process outright (no exception, no traceback). Batches of 4, 8, 12 and 16 were
fine. `_forward()` now chunks at 8, which is a working guard rather than a
diagnosis — the underlying limit in OpenCV 5.0.0.93 is unknown.

## 🟡 3. Face detection failure is silent

When YuNet finds no face, the whole frame is analysed and the verdict is
returned as if a face had been found. The user is never told. For non-face
images this produces confident, meaningless output.

**Fix:** return a `faceFound: false` flag and have the result page say so.

## 🟡 4. Verifier reliability is calibrated on very little data

When the verifiers are enabled (`DS_VERIFIERS=1`), their weights and the
"all ≥ 0.85 to overrule" rule were tuned against roughly ten images. Measured
misbehaviour that motivated the current settings:

- SigLIP scored a **re-saved authentic portrait at 1.00 fake**
- ViT scored the same authentic portrait at 0.67 fake
- SigLIP scores 0.00 on StyleGAN faces — it detects diffusion images, so its
  silence is not evidence of authenticity

The default (verifiers off) avoids this entirely, and our own model scores
9/9 alone. Anyone turning them on should re-validate first.

## 🟡 5. Test accuracy is missing from the shipped model metadata

`deepshield.onnx.json` has `"test_accuracy": null`. The 140k test split was
never scored for V3: the Kaggle session was interrupted and the export was
rebuilt from the resume checkpoint, which is written before the test cell
runs. Validation, robust and TPDN figures are present and real.

**Fix:** run the evaluation cell against `v3_max.pth` and patch the JSON.

## 🟡 6. Training curves for V3 cannot be reproduced

The per-epoch history exists inside the resume checkpoint, but the plotted
`training_curves.png` in `training/results/` is from **V2**. The V3 numbers
are recorded in `MODEL_CARD.md` and were printed live during training.

**Fix:** re-plot from the history array in the resume file, if it is still
available.

## 🟡 7. Kaggle notebooks do not survive a lost session

The Colab notebook checkpoints to Google Drive and resumes cleanly. The Kaggle
version writes to `/kaggle/working`, which is lost when an interactive session
ends without saving a version. The V4 run was cancelled and its progress could
not be recovered.

**Fix:** use *Save Version → Save & Run All (Commit)* for long runs, which
executes headless and persists outputs.

## 🟡 8. Hugging Face downloads stall on some networks

Fetching the verifier models repeatedly froze at ~245 MB. The cause is HF's
Xet transfer protocol; setting `HF_HUB_DISABLE_XET=1` falls back to plain
HTTP and completes at full speed. Documented in the README troubleshooting
table.

## 🟡 9. `DOCUMENTATION.md` predates most of the system

`docs/DOCUMENTATION.md` was written when the project was frontend-only. It
carries a note about the folder restructure, but nothing about the backend,
the model, the ensemble, explainability, the landing page or the ONNX
migration. `CURRENT_STATE.md` and `MODEL_CARD.md` supersede it for anything
factual.

## 🟡 10. Feedback data has no operational path

`POST /api/feedback` appends to `data/feedback.jsonl`, which is gitignored and
never read by anything. The dashboard's "user-rated correct" figure is
computed from the browser's own copy in `localStorage`, so the server-side
file currently accumulates without a consumer.

## 🔵 11. Face-swap deepfakes are not detected

The headline limitation. A real DFDC video scored **97% "real"**. V3 learned
fully generated faces; face-swaps are a different artefact family. Expected
behaviour for this model, not a regression — see `MODEL_CARD.md`.

## 🔵 12. Cross-generator generalisation is not guaranteed

V2 scored thispersondoesnotexist faces at 0.02–0.49 until StyleGAN2 was added
to training; V3 scores the same images 0.97–0.98. A future generator can open
the same gap again. Mitigated by multi-generator training and the optional
verifiers, not solved.

## 🔵 13. Processed media degrades detection

Screenshots, repeated compression and platform re-encoding destroy the
artefacts detection depends on. This is why streaming-platform URLs are
refused with an explanation rather than downloaded. Affects every detector
on the market.

## 🔵 14. Fairness is unmeasured

No evaluation has been run across skin tone, age or gender. The training real
class is FFHQ-only. Nothing here should be read as a fairness claim.

---

## Fixed, kept for the record

| Issue | Resolution |
|---|---|
| V1 learned the dataset's JPEG/resize fingerprint | Anti-shortcut augmentation + robust-based checkpoint selection (V2) |
| StyleGAN2 faces scored 0.02–0.49 | Multi-generator training (V3): the same images now score 0.97–0.98 |
| A 2687px authentic portrait scored 0.94 fake | Inputs capped at 1024px |
| A pristine camera original scored 0.95 fake | JPEG q88 round-trip before inference |
| Averaging buried a confident specialist | Own-led weighting with a corroboration rule |
| Twelve Flask instances accumulated in one session | `scripts/ds.js` works on the port; debug reloader off by default |
| Clearing history left the last report visible | `Clear history` now also clears the session scan |
| Reports page showed only the newest report | "All reports" list added |
| `hidden` attribute lost to `.empty-state { display: flex }` | Global `[hidden] { display: none !important }` |
| Backend needed 1.9 GB to run a 17 MB model | ONNX through OpenCV; 197 MB, verified identical to 3e-7 |
| `/api` reported MobileNetV3-Small / 2.5M / PyTorch for a Large ONNX model | `MODEL_INFO` deleted; identity comes from the model's own metadata (Phase 1), and `scripts/model_test.py` now asserts the file, the engine and the API agree |
| Any URL could be fetched, including `127.0.0.1` and cloud metadata | Scheme, DNS and per-redirect address validation (Phase 2) |
| Uploads were accepted on extension alone | Size, MIME, magic bytes, decoder, dimensions and duration (Phase 2) |
| Nothing limited request rate or concurrency | 5/min per client, 2 concurrent analyses, 30-minute upload sweep (Phase 2) |
| The model reported no version | Identity block — `model_name`, `architecture`, `version`, `runtime`, `input_size` (Phase 3) |
