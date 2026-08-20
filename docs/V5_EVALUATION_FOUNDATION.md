# DeepShield V5 evaluation and calibration foundation

This is an evaluation protocol, not a new model. It does not download data,
train, export, replace, or load a new production ONNX model.

## Manifest

Use `tools/dataset_manifest.py` to build the one evaluation manifest. Every
media row records `relative_path` (the safe media path), `label`, `modality`,
`source_dataset`, `identity_group` (or legacy `identity_id`),
`manipulation_type`, `generator_family`, `compression_slice`,
`robustness_slice`, `split`, and provenance. The valid splits are:

- `train`
- `calibration`
- `validation`
- `sealed_test`

`robustness_slice` is one of `clean`, `jpeg`, `resize`, `blur`, `screenshot`,
or `low_light`. `generator_disjoint=yes` on a fake sealed-test row means that
its generator family must not occur in any non-sealed split.

The builder and checker reject, with row-specific errors:

- absolute, parent-traversal, escaping, or symlink media paths;
- missing source/group/split/provenance facts;
- a media path or content hash present in more than one split;
- an identity/original-video group present in more than one split;
- generator-family leakage into a generator-disjoint sealed test.

Invalid manifests are still written as an audit ledger but the CLI exits with
status 2; they cannot be treated as a passing evaluation.

## Commands

```powershell
# Build and validate the manifest. metadata.csv supplies custodian facts;
# no label is inferred from a filename or directory.
venv\Scripts\python.exe tools\dataset_manifest.py `
  --dataset approved-data\v5 `
  --metadata approved-data\v5\metadata.csv `
  --out approved-data\v5_manifest.csv

# Revalidate the immutable manifest before scoring.
venv\Scripts\python.exe tools\dataset_manifest.py `
  --dataset approved-data\v5 --out approved-data\v5_manifest.csv --validate

# Score through the current serving engine. It writes p_fake as an
# uncalibrated model score; it does not alter the production model.
venv\Scripts\python.exe tools\benchmark_model.py `
  --dataset approved-data\v5 --out approved-data\v5_predictions

# A. Fit temperature on calibration only. This writes no thresholds or report.
venv\Scripts\python.exe scripts\evaluate.py `
  --from-csv approved-data\v5_predictions\predictions.csv `
  --manifest approved-data\v5_manifest.csv `
  --dataset-root approved-data\v5 `
  --model-artifact models\deepshield.onnx `
  --fit-temperature --split calibration `
  --calibration-out approved-data\v5_temperature.json

# B. Apply frozen calibration, select the inconclusive band on validation,
# and write a separate threshold artifact.
venv\Scripts\python.exe scripts\evaluate.py `
  --from-csv approved-data\v5_predictions\predictions.csv `
  --manifest approved-data\v5_manifest.csv `
  --dataset-root approved-data\v5 `
  --model-artifact models\deepshield.onnx `
  --calibration-in approved-data\v5_temperature.json `
  --select-thresholds --split validation `
  --thresholds-out approved-data\v5_thresholds.json

# C. Evaluate sealed_test with both frozen artifacts. This cannot fit, select,
# or overwrite either artifact.
venv\Scripts\python.exe scripts\evaluate.py `
  --from-csv approved-data\v5_predictions\predictions.csv `
  --manifest approved-data\v5_manifest.csv `
  --dataset-root approved-data\v5 `
  --model-artifact models\deepshield.onnx `
  --calibration-in approved-data\v5_temperature.json `
  --thresholds-in approved-data\v5_thresholds.json `
  --split sealed_test --json-report approved-data\v5_sealed_report.json
```

The same stages use slash-separated paths on Linux/GitHub Actions:

```bash
python scripts/evaluate.py --from-csv /data/v5_predictions/predictions.csv \
  --manifest /data/v5_manifest.csv --dataset-root /data/v5 \
  --model-artifact models/deepshield.onnx --fit-temperature \
  --split calibration --calibration-out /data/v5_temperature.json

python scripts/evaluate.py --from-csv /data/v5_predictions/predictions.csv \
  --manifest /data/v5_manifest.csv --dataset-root /data/v5 \
  --model-artifact models/deepshield.onnx \
  --calibration-in /data/v5_temperature.json --select-thresholds \
  --split validation --thresholds-out /data/v5_thresholds.json

python scripts/evaluate.py --from-csv /data/v5_predictions/predictions.csv \
  --manifest /data/v5_manifest.csv --dataset-root /data/v5 \
  --model-artifact models/deepshield.onnx \
  --calibration-in /data/v5_temperature.json \
  --thresholds-in /data/v5_thresholds.json --split sealed_test \
  --json-report /data/v5_sealed_report.json
```

`--model-id` can replace `--model-artifact` when the checkpoint file is not
available on the evaluation host. Use the same identity mode in all stages.

The V5 JSON report includes precision, recall, specificity, F1, ROC-AUC,
PR-AUC, FPR, FNR, confusion matrix, per-dataset/per-manipulation metrics,
all required robustness-slice keys, Brier, ECE, reliability-diagram data, and
group-aware bootstrap confidence intervals. Empty required slices are marked
as not measured rather than silently omitted.

## Calibration and decisions

Temperature scaling accepts `logit_fake_minus_real`/`log_odds_fake`, or a pair
of `fake_logit` and `real_logit` columns. A lone class logit is rejected because
it is not a binary log-odds value. If a prediction CSV has only a score in
`[0,1]`, it safely clips and transforms the
binary score to log-odds before fitting. Calibration and threshold artifacts
record a schema version, UTC creation time, source split, fake-positive class
convention, model identifier/SHA-256, and manifest SHA-256. Thresholds also
bind to the exact calibration-artifact hash. A later stage rejects any mismatch
instead of silently applying stale artifacts. The validation-only policy
returns:

```text
score <= real_threshold  -> real
score >= fake_threshold  -> fake
otherwise                -> inconclusive
```

The minimum abstention band defaults to 0.10 score units. Sealed-test rows are
not passed to either fitting function.

## Issue #26 gate

This foundation implements the local protocol portions of Issue #26. The
following still require authorised data and a later experiment: access/terms
for FF++, DFDC, Celeb-DF v2 and DeeperForensics; redistribution permission for
derived weights; consented modern real phone media; sealed unseen AI image and
video families; fairness metadata where allowed; and deployment-target latency
and RSS measurements. See [the V4 notebook audit](V5_V4_NOTEBOOK_AUDIT.md).
