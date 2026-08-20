# DeepShield V5 audit — `DeepShield_V4_Universal.ipynb`

Audit date: 2026-08-20. This is a static audit of the notebook source; it was
not executed, no dataset was downloaded, and no model was trained or replaced.

## Result

`DeepShield_V4_Universal.ipynb` is not eligible to train or promote a V5
model as written. It contains useful DFDC group-split work, but it does not
meet Issue #26's sealed-evaluation, preprocessing-parity, calibration, or
deployment-evidence gate. Do not infer a universal-detection result from its
`V4-Universal` checkpoint metadata.

## Findings

| Severity | Finding | Evidence in notebook | Why it blocks a V5 claim |
|---|---|---|---|
| Major | Production preprocessing drift | Cell 4 uses `Resize -> ToTensor -> Normalize` for `eval_tf`; `robust_tf` adds only JPEG q40. | Production caps the original frame, runs YuNet face detection/cropping, applies a q88 JPEG round trip, normalizes, and TTA-averages mirrored inputs. The notebook evaluates raw/pre-cropped images without that serving path. |
| Major | DFDC input-domain drift | Cell 2 consumes one pre-extracted 224px crop per DFDC video; Cell 4 scores it directly. | Deployment samples frames and detects faces at runtime. A result on pre-extracted crops is not video or serving-path performance. |
| Major | Holdouts used for checkpoint selection | Cell 7 chooses `best_state` from `robust_acc`, `tpdn_acc`, and `dfdc_acc` every epoch. | TPDN and DFDC “holdouts” are tuning data, not sealed tests. Their final score is optimistically selected. A derived JPEG-q40 copy of validation is also part of selection. |
| Major | Missing calibration protocol | Cells 7–10 use softmax `p_fake` and a fixed 0.5 decision only. | There is no calibration split, temperature fit, ECE/Brier/reliability result, or validation-only fake/real/abstention policy. Scores cannot be called probabilities. |
| Major | Generator-disjointness is not auditable | Cells 3–5 remove the string `UNSEEN_FAMILY` from `gen_sources`, but record no family-level manifest or seal. | It cannot prove that aliases, derived images, or another source in the same generator family did not enter training. If the optional diffusion input is absent, the unseen test is empty and merely logged. |
| Major | No Issue #26 sealed real-world test | The only final sets are 140k test, TPDN, optional DFDC, and optional diffusion (Cell 9). | Celeb-DF v2 and DeeperForensics-1.0 are absent; there is no consented phone-photo sealed real set, sealed fully-AI image/video set, or cross-dataset control of real-media FPR. |
| Moderate | Incomplete leakage validation | Cell 5 checks exact path overlap. The DFDC identity check is good when `metadata.csv` has `original`, but Cell 2 permits a per-video fallback. It explicitly trusts the 140k packaged split. | Exact paths do not detect copies, renamed files, symlinks, or group/identity reuse outside DFDC. The fallback can put related original/fake identities in different groups. |
| Moderate | No manifest-level split ledger | Samples are held in in-memory lists (Cell 4). | There is no immutable record of media path, provenance, group, manipulation, generator family, modality, robustness slice, and split. Results cannot be reproduced or audited after a Kaggle session. |
| Moderate | Robustness scope is too narrow | Cell 4 defines only q40 JPEG for `robust_tf`. | Issue #26 requires JPEG/re-encode, resize, blur, screenshots, low light, and low resolution slices. One derived JPEG variant cannot establish this. |
| Moderate | No fairness, latency, or memory evidence | No notebook cell measures demographic slices, Render CPU latency, peak RSS, or a one-worker 512 MB run. | A model can improve accuracy and still regress false positives or fail the deployment envelope. |
| Moderate | Resume state mixes an experiment with a mutable working directory | Cell 7 loads `/kaggle/working/resume.pth` when present. | Without a manifest hash and split/seal fingerprint, a resumed state can silently correspond to a different dataset inventory or split. |
| Minor | “Universal” naming is unsupported | Cell 10 writes `version: 'V4-Universal'`; Cell 2 optionally downloads sources and Cell 9 reports a few image/crop sets. | The tested scope excludes sealed external datasets, fully-AI video, and real-world phone-photo coverage. The name encourages a claim the evidence cannot support. |

## What is worth preserving

- The DFDC original-video grouping in Cells 2 and 4 is the right direction.
  When `original` metadata exists, the notebook correctly keeps all related
  real/fake files in one group.
- The Cell 5 exact-path and DFDC-group assertions identify an important
  historical leakage mode.
- `UNSEEN_FAMILY` is deliberately excluded from Cell 7's checkpoint score.
  That is correct, but it does not turn the other selected holdouts into
  sealed tests.
- Cell 9 exports raw prediction rows rather than hand-writing metrics. V5
  should retain that idea, but attach each row to the V5 manifest.

## Required changes before any V5 training experiment

1. Create and validate the V5 manifest before data preparation. Give every
   sample a safe relative path, label, manipulation/generator metadata,
   dataset, identity/original-video group, modality, robustness slice, and
   one of `train`, `calibration`, `validation`, or `sealed_test`.
2. Use `training/deepshield_preprocess.py` for deterministic validation and
   sealed scoring. Training augmentation may precede its production q88 tail;
   evaluation must not use a separate transform.
3. Keep calibration, validation, and sealed-test groups disjoint. Fit only
   temperature on calibration; select fake/real/inconclusive thresholds only
   on validation; evaluate sealed data without changing either.
4. Lock generator-disjoint sealed sets at the generator-family level, not a
   notebook variable name. Treat an empty proposed holdout as a failed gate.
5. Do not use DFDC or TPDN as checkpoint-selection signals if they are meant
   to support a final claim. Reclassify them as validation, or make their
   groups sealed and remove them from Cell 7's selection score.
6. Add the Issue #26 data/access prerequisites and measure controlled FPR,
   robustness, fairness where permitted, latency, and peak memory before
   considering a production promotion.

## Issue #26 status after this implementation

The local foundation now supplies the manifest and metric/calibration guard
rails needed for the scientific gate. It does **not** satisfy the issue's
dataset-access, consent, derived-weight redistribution, held-out media,
fairness metadata, latency, or memory requirements. Those require authorised
data acquisition and a later, separately reviewed experiment.
