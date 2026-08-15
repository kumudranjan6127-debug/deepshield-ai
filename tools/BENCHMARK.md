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

Results are written to `benchmark-results/predictions.csv` and `benchmark-results/report.json`.

## Metrics

The report includes accuracy, precision, recall, F1, false-positive rate, false-negative rate, no-face/inconclusive count, and latency. **False negatives are especially important:** these are manipulated samples incorrectly reported as real.

## Recommended evaluation matrix

Run the same script for each checkpoint/model version and preserve the resulting JSON. At minimum, evaluate:

1. authentic camera photos;
2. FaceForensics++/DFDC-style manipulated faces;
3. a held-out deepfake set;
4. modern AI-generated faces/images as a separate category.

The current script intentionally uses only `real` and `fake` labels because the existing classifier is binary. AI-generated media should be reported separately in a future multi-class/ensemble benchmark rather than silently treating it as equivalent to a face-swap.
