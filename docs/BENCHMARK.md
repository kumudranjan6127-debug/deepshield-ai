# DeepShield Benchmark

> **Read this first.** The table below is generated, not typed:
>
> ```bash
> python scripts/evaluate.py --data eval_data --report docs/BENCHMARK_TABLE.md
> ```
>
> Every number is reproducible from `eval_data/predictions.csv`, the file
> that produced it. Where a number does not exist, this document says so and
> names exactly what would produce it. **Nothing here is estimated.**

| | |
|---|---|
| Model | **DeepShield V3-Max** — MobileNetV3-Large, ONNX, 224×224 |
| Classes | `["fake", "real"]`, index order fixed |
| Machine | Windows 10, 4 CPU cores, **CPU only**, no GPU |
| Generated | 2026-08-11 |

---

## 1. Detection quality

**These numbers are real and they are also nearly meaningless. Read the
second row before the first.**

| Metric | Value | Basis |
|---|---|---|
| Images scored | **92** | 24 real, 68 fake |
| **Independent groups** | **12** | the honest sample size — images inside a group are correlated |
| Accuracy | **100.00%** | at threshold 0.50 |
| Precision | **100.00%** | of everything called fake |
| Recall | **100.00%** | of the fakes present |
| F1 | **100.00%** | |
| Specificity | **100.00%** | of the real images |
| ROC-AUC | **1.0000** | ranking quality |
| PR-AUC | **1.0000** | |
| **False-positive rate** | **0.00%** | **a real photograph called fake** |
| False-negative rate | **0.00%** | a deepfake called real |
| Brier score | **0.0006** | calibration, 0 is perfect |
| ECE | **0.0240** | calibration error |
| Cross-dataset accuracy | *not measured* | no generator outside the training families is present |
| DFDC (face-swap) | *not measured* | see §3 |

### Why 100% is not a result

The real class is **one person**. Twenty-four frames of one recording is one
independent observation, not twenty-four. The fake class is five StyleGAN2
stills, their processed variants, and one clip — twelve groups in total.

A detector that scores 100% on twelve groups has told you almost nothing. It
has ruled out being *broken*; it has not demonstrated being *good*. The
project's own harness prints the group count next to the sample size for
exactly this reason.

What this does establish, narrowly:

- the pipeline runs end to end and the two classes are separated by a wide
  margin (mean P(fake) 0.975 vs 0.022)
- detection survives phone, screenshot, messaging-app and re-encode
  processing on this sample (§2)
- **no false positive was produced on the one authentic subject available**

### Per source

| Class | Source | n | Mean P(fake) | Outcome |
|---|---|---|---|---|
| fake | `tpdn` | 5 | 0.974 | 100.00% detected |
| fake | `orig` | 5 | 0.975 | 100.00% detected |
| fake | `phone` | 5 | 0.974 | 100.00% detected |
| fake | `screenshot` | 5 | 0.974 | 100.00% detected |
| fake | `social` | 5 | 0.975 | 100.00% detected |
| fake | `reencode` | 5 | 0.975 | 100.00% detected |
| fake | `synthetic_clip` | 38 | 0.976 | 100.00% detected |
| real | `portrait` | 24 | 0.022 | **0.00% called fake** |

---

## 2. Robustness to processing

Same five StyleGAN2 faces through each condition. The files genuinely
differ — 277 KB, 44 KB and a 1.3 MB PNG for one image — yet the score moves
by at most **0.006**.

| Condition | What was done | Detected | Mean P(fake) |
|---|---|---|---|
| original | q95 re-save | 5 / 5 | 0.975 |
| phone | 1440px cap, q92 | 5 / 5 | 0.974 |
| screenshot | 1080px cap, PNG | 5 / 5 | 0.974 |
| social | 720px cap, q60 | 5 / 5 | 0.975 |
| re-encode | q55 then q40 | 5 / 5 | 0.975 |

That flatness is the compression normalisation and the 224px face crop doing
their job. Both were added to *stop false positives*; robustness is a side
effect.

---

## 3. What has never been measured

These are blank because no data exists to fill them, not because they were
forgotten. Each row names the input that would produce it.

| Missing | Why it matters | What would produce it |
|---|---|---|
| **False-positive rate at scale** | The expensive error. One authentic subject is not a rate | A few hundred genuine photographs in `eval_data/real/photos` |
| **Cross-dataset accuracy** | In-domain accuracy says how well a model memorised the generators it saw. This says what a user gets | A generator outside the training families, then `--seen sg1,sg2,tpdn,diffusion` |
| **DFDC / face-swap** | The headline limitation: a real DFDC video scored **97% "real"** | DFDC face crops in `eval_data/fake/dfdc` |
| **Calibration at scale** | The percentage the UI shows carries no probabilistic promise | The same labelled set; `evaluate.py` already prints ECE, MCE, Brier and a reliability diagram |
| **Fairness** | The training real class was FFHQ only | A labelled set with demographic annotations |
| **FF++ / Celeb-DF** | Comparability with published work | Those datasets, scored through `evaluate.py` |

Filling the first three is roughly an afternoon each. `eval_data/README.md`
gives the folder layout and the download links.

---

## 4. Latency

`python scripts/benchmark.py` — median of 3 runs, stages disjoint (the
forwards inside the heatmap are billed to `explain`, not counted twice).

| Case | decode+detect | normalise | prepare | forward | explain | other | **total** |
|---|---|---|---|---|---|---|---|
| image 224px | 0.004 | 0.002 | 0.004 | 0.018 | **0.390** | — | **0.42 s** |
| image 1024px | 0.072 | 0.016 | 0.017 | 0.019 | **0.377** | — | **0.49 s** |
| video 10 s | 0.139 | 0.028 | 0.048 | 0.186 | — | 0.093 | **0.49 s** |
| video 30 s | 0.447 | 0.086 | 0.166 | 0.599 | — | 0.266 | **1.57 s** |
| video 60 s | 0.890 | 0.174 | 0.291 | 1.179 | — | 0.453 | **2.99 s** |

**Average image latency: ~455 ms. Video: ~50 ms per sampled frame**, flat at
every length — 0.5 s for 10 seconds of footage, 3.0 s for 60.

**Images are the explanation.** Occlusion sensitivity is **93%** of a 224px
image's latency and 76% of a 1024px one — 36 extra forward passes to explain
a verdict the model itself reached in 18 ms. Batch size is not a lever: 36
forwards take 320–331 ms whether they go through 4, 8, 12, 16 or 18 at a
time.

Benchmark clips are 480×480 at 10 fps. A phone shoots 720p at 30 fps, where
frame decoding costs more — read the video figures as a floor.

---

## 5. Resources

| Case | frames | peak RSS | over baseline | CPU s/run | cores used |
|---|---|---|---|---|---|
| image 224px | 1 | 184 MB | 43 MB | 0.92 | 2.2 |
| image 1024px | 1 | 260 MB | 52 MB | 1.17 | 2.4 |
| video 10 s | 10 | 227 MB | 13 MB | 1.11 | 2.3 |
| video 60 s | 60 | 230 MB | 16 MB | 6.09 | 2.0 |

Peak RSS **260 MB**, inside a 512 MB hosting tier. Backend dependencies are
197 MB and the model is 16.8 MB, because it runs as ONNX through OpenCV and
needs no PyTorch at runtime.

---

## 6. Correctness suites

Not accuracy — the properties that must hold whatever the accuracy turns out
to be. `python -m pytest`, ~20 s, no server and no network.

| Suite | Tests |
|---|---|
| `test_api.py` | 25 |
| `test_upload.py` | 21 |
| `test_validation.py` | 22 |
| `test_security.py` | 76 |
| `test_inference.py` | 29 |
| `test_video.py` | 36 |
| `test_model_parity.py` | 25 |
| `test_metrics.py` | 19 |
| `test_split.py` | 8 |
| **total** | **261** |

Plus `scripts/regression_test.py verify` — 24 recorded values, diffed against
a baseline, which answers "did this change alter any behaviour".

---

## 7. Reproducing this

```bash
# 1. put labelled images in eval_data/  (see eval_data/README.md)
python scripts/evaluate.py --data eval_data --report docs/BENCHMARK_TABLE.md

# 2. in-domain vs a generator the model never saw
python scripts/evaluate.py --seen sg1,sg2,tpdn,diffusion

# 3. what a 1% false-positive budget costs in recall
python scripts/evaluate.py --target-fpr 0.01

# 4. latency and memory
python scripts/benchmark.py --repeats 5
```

A Kaggle training run writes `predictions_all.csv`; `--from-csv` turns it
into the same table, computed by the same arithmetic. No figure in this
project is produced by one implementation and published by another.
