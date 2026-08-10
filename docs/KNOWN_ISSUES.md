# Known Issues

Every entry below has been **observed**, not suspected. Where a number is
given, it was measured. Last reviewed after Phase 5, 2026-08-10.

Severity: 🔴 wrong output or wrong information · 🟡 works but flawed ·
🔵 limitation to document rather than fix

---

## 🟡 1. The false-positive rate has never been measured

Every figure the project reports is accuracy: 99.90% validation, 99.18%
robust, 100% TPDN. None of them say how often an **authentic photograph is
called fake**, which is the error that actually costs someone something.

Measuring it needs genuine photographs scored through the deployed
pipeline, and this repository has none — the training real class was FFHQ,
which lives on Kaggle. Two false positives have been found by hand (a
2687px portrait at 0.94, a camera original at 0.95); both were fixed, but
by inspection rather than measurement.

The tooling now exists and the missing piece is only data.

**Fix:** fill `eval_data/real/photos` and run
`python scripts/evaluate.py --target-fpr 0.01`. See `eval_data/README.md`.

## 🟡 2. Calibration has never been measured

The UI shows a percentage. Nothing has ever checked whether that
percentage behaves like one — whether images scored 90 are fake about 90%
of the time. Expected Calibration Error, Brier score and a reliability
curve are all unknown for V3.

This matters because a network trained with cross-entropy and selected on
validation accuracy is usually **over-confident**: it will report 0.97 on
evidence worth 0.80. That is why Phase 5 replaced probability language
with evidence language, and why `/api/health` reports
`"calibrated": false`.

`scripts/evaluate.py` prints ECE, MCE, Brier and a reliability diagram for
any labelled set. It needs the same data as [issue 1](#1).

**Fix:** measure it. If the model turns out over-confident, temperature
scaling on a held-out split is a few lines and does not require retraining.

## 🟡 3. The lowest certainty band can never be reached

`CERTAINTY_BANDS` spans 0-100, but `confidence` is `max(p, 1-p) * 100` for
two classes, so it is **never below 50**. "Low evidence" (0-30) is
unreachable by construction, and "Uncertain" (30-70) can only ever hold
the top half of its range. Proven, not suspected —
`scripts/metrics_test.py` scores 4,000 random predictions and the bottom
band stays empty.

The band table was specified against `confidence`. Two ways out, both
deferable until there is data:

- keep four bands and move the cut points into 50-100, or
- band on evidence strength, `|2p - 1| * 100`, which does span 0-100.

**Fix:** pick one once `scripts/evaluate.py` has measured accuracy per
band on real data — the occupancy column makes the answer obvious.

## 🟡 4. `ds_feedback` is not registered in `DS.KEYS`

`DS.KEYS` lists `ds_user`, `ds_scan`, `ds_history`, `ds_settings`. The
feedback store is written as a raw string in three places
(`pages/result.js` ×2, `pages/dashboard.js` ×1). It works, but it sits outside
the convention every other key follows, and "Clear history" does not clear it.

**Fix:** add `FEEDBACK: 'ds_feedback'` to `DS.KEYS` and use it.

## 🟡 5. OpenCV's DNN engine crashes on large batches

Discovered while building the occlusion heatmap: a 36-image batch killed the
process outright (no exception, no traceback). Batches of 4, 8, 12 and 16 were
fine. `_forward()` now chunks at 8, which is a working guard rather than a
diagnosis — the underlying limit in OpenCV 5.0.0.93 is unknown.

## 🟡 6. Face detection failure is silent

When YuNet finds no face, the whole frame is analysed and the verdict is
returned as if a face had been found. The user is never told. For non-face
images this produces confident, meaningless output.

**Fix:** return a `faceFound: false` flag and have the result page say so.

## 🟡 7. Verifier reliability is calibrated on very little data

When the verifiers are enabled (`DS_VERIFIERS=1`), their weights and the
"all ≥ 0.85 to overrule" rule were tuned against roughly ten images. Measured
misbehaviour that motivated the current settings:

- SigLIP scored a **re-saved authentic portrait at 1.00 fake**
- ViT scored the same authentic portrait at 0.67 fake
- SigLIP scores 0.00 on StyleGAN faces — it detects diffusion images, so its
  silence is not evidence of authenticity

The default (verifiers off) avoids this entirely, and our own model scores
9/9 alone. Anyone turning them on should re-validate first.

## 🟡 8. Test accuracy is missing from the shipped model metadata

`deepshield.onnx.json` has `"test_accuracy": null`. The 140k test split was
never scored for V3: the Kaggle session was interrupted and the export was
rebuilt from the resume checkpoint, which is written before the test cell
runs. Validation, robust and TPDN figures are present and real.

The same gap is wider than one field: no precision, recall, F1,
specificity, ROC-AUC, PR-AUC or FPR has ever been computed for V3.

**Fix:** score `v3_max.pth` with `scripts/evaluate.py` and patch the JSON
from the result.

## 🟡 9. Training curves for V3 cannot be reproduced

The per-epoch history exists inside the resume checkpoint, but the plotted
`training_curves.png` in `training/results/` is from **V2**. The V3 numbers
are recorded in `MODEL_CARD.md` and were printed live during training.

**Fix:** re-plot from the history array in the resume file, if it is still
available.

## 🟡 10. Kaggle notebooks do not survive a lost session

The Colab notebook checkpoints to Google Drive and resumes cleanly. The Kaggle
version writes to `/kaggle/working`, which is lost when an interactive session
ends without saving a version. The V4 run was cancelled and its progress could
not be recovered.

**Fix:** use *Save Version → Save & Run All (Commit)* for long runs, which
executes headless and persists outputs.

## 🟡 11. Hugging Face downloads stall on some networks

Fetching the verifier models repeatedly froze at ~245 MB. The cause is HF's
Xet transfer protocol; setting `HF_HUB_DISABLE_XET=1` falls back to plain
HTTP and completes at full speed. Documented in the README troubleshooting
table.

## 🟡 12. `DOCUMENTATION.md` predates most of the system

`docs/DOCUMENTATION.md` was written when the project was frontend-only. It
carries a note about the folder restructure, but nothing about the backend,
the model, the ensemble, explainability, the landing page or the ONNX
migration. `CURRENT_STATE.md` and `MODEL_CARD.md` supersede it for anything
factual.

## 🟡 13. Feedback data has no operational path

`POST /api/feedback` appends to `data/feedback.jsonl`, which is gitignored and
never read by anything. The dashboard's "user-rated correct" figure is
computed from the browser's own copy in `localStorage`, so the server-side
file currently accumulates without a consumer.

## 🔵 14. Face-swap deepfakes are not detected

The headline limitation. A real DFDC video scored **97% "real"**. V3 learned
fully generated faces; face-swaps are a different artefact family. Expected
behaviour for this model, not a regression — see `MODEL_CARD.md`.

## 🔵 15. Cross-generator generalisation is not guaranteed

V2 scored thispersondoesnotexist faces at 0.02–0.49 until StyleGAN2 was added
to training; V3 scores the same images 0.97–0.98. A future generator can open
the same gap again. Mitigated by multi-generator training and the optional
verifiers, not solved.

## 🔵 16. Processed media degrades detection

Screenshots, repeated compression and platform re-encoding destroy the
artefacts detection depends on. This is why streaming-platform URLs are
refused with an explanation rather than downloaded. Affects every detector
on the market.

## 🔵 17. Fairness is unmeasured

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
| The V4 notebook split DFDC by filename | Several fakes share one source video, and that video is itself in the real class — on synthetic data with the same structure, **78 of 100** held-out files shared a face with training. The split is now by identity group, asserted at runtime and by `scripts/split_test.py` (Phase 4) |
| Nothing in the repo could compute a metric | `ds_metrics.py` + `evaluate.py`: accuracy, precision, recall, F1, specificity, ROC-AUC, PR-AUC, FPR, FNR, per-source, threshold sweep — verified by 40 known-answer tests (Phase 4) |
| `security_test.py` aborted on a plain Windows console | A check was named with a `→`, which cp1252 cannot encode; the suite died mid-run. Printed strings are ASCII now (Phase 4) |
| The UI called the heatmap "Grad-CAM attention" | It is occlusion sensitivity, and it measures how much the prediction moves when a region is hidden — not where the network attends. Renamed everywhere it was shown (Phase 5) |
| A confidence percentage read as a probability | The verdict now carries a `certainty` band, and the wording says "detection confidence", with the bands published by `/api/health` so no threshold lives in the browser (Phase 5) |
