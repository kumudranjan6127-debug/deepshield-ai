# DeepShield model benchmark

This benchmark evaluates the installed **live** model without using filenames as a signal.

## Dataset layout

```text
dataset/
  real/
    camera_001.jpg
    photo_002.jpg
  fake/
    manipulated_001.jpg
    deepfake_002.jpg
```

Keep identities disjoint between training and evaluation. Do not put the same person's frames/images in both sets. For a meaningful benchmark, also keep a separate external holdout set that was not used for training or threshold tuning.

## Run

From the repository root:

```bash
python tools/benchmark_model.py --dataset dataset
```

Results are written to:

- `benchmark-results/predictions.csv` — one row per input, including label,
  verdict, confidence, face detection result/count, latency, and any error.
- `benchmark-results/report.json` — model identity, protocol, metrics, and
  latency statistics for automated comparison.
- `benchmark-results/summary.txt` — a concise human-readable report.

The runner calls `inference.analyze_file(path, "image")`, the same public
serving entry point used by the backend. It only supplies the path to load
pixels; neither file names nor paths are passed to the detector or model.
No-face inputs are retained in the per-file output and counted as
inconclusive; they are excluded from classification metrics because the
face-trained model explicitly reports insufficient evidence for them.

`dataset/` and `benchmark-results/` are ignored by Git. Keep the source
dataset outside version control and retain the generated JSON/CSV with the
dataset provenance needed to reproduce a reported result.

## Metrics

The report includes accuracy, precision, recall, F1, false-positive rate,
false-negative rate, no-face/inconclusive count, errors, and mean/median/p95
latency. **False negatives are especially important:** these are manipulated
samples incorrectly reported as real. A metric with no applicable samples is
written as `null`/`n/a`, never fabricated as zero.

## Recommended evaluation matrix

Run the same script for each checkpoint/model version and preserve the resulting JSON. At minimum, evaluate:

1. authentic camera photos;
2. FaceForensics++/DFDC-style manipulated faces;
3. a held-out deepfake set;
4. modern AI-generated faces/images as a separate category.

The current script intentionally uses only `real` and `fake` labels because the existing classifier is binary. AI-generated media should be reported separately in a future multi-class/ensemble benchmark rather than silently treating it as equivalent to a face-swap.
