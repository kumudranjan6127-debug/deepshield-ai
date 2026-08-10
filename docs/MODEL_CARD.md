# Model Card — DeepShield V3-Max

| | |
|---|---|
| **Name** | DeepShield V3-Max |
| **Architecture** | **MobileNetV3-Large** (torchvision), classifier head replaced with a 2-class linear layer |
| **Format in production** | **ONNX** (`models/deepshield.onnx`, 16.8 MB), executed by OpenCV's DNN module |
| **Input** | RGB image, **224 × 224**, ImageNet normalisation (mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`) |
| **Output** | 2 classes — **`["fake", "real"]`**, index order fixed; softmax probabilities |
| **Parameters** | ~5.4 M |
| **Precision** | float32 |
| **Device** | CPU only |
| **Trained** | 2026-08-08, Kaggle, single T4 GPU, 10 epochs |
| **Source checkpoint** | `models/deepshield_mobilenetv3.pth` (archived as `v3_max.pth`) |
| **Card updated** | 2026-08-10, commit `ab0103d` |

---

## Intended use

Screening images (and, frame by frame, video) for **fully AI-generated human
faces** — the kind used for fake profiles and fabricated "photo evidence".

Designed to run on ordinary hardware: the whole backend is ~197 MB and peaks
around 200 MB of RAM, so it needs no GPU and no cloud service.

**Not intended for**, and not validated for:
- forensic or legal evidence
- deciding anything about a person on its own
- images without a clearly visible face
- audio, text, or non-face imagery

---

## Training data

| Source | Class | Role |
|---|---|---|
| 140k Real and Fake Faces (`xhlulu`) — StyleGAN1 fakes | fake | Base family |
| thispersondoesnotexist set (`almightyj`) — StyleGAN2 | fake | Added in V3 |
| Fakefaces (`hyperclaw79`) — StyleGAN2 hi-res | fake | Added in V3 |
| Stable Diffusion faces (`mohannadaymansalah`) | fake | Added in V3 |
| 140k dataset real split (FFHQ / Flickr portraits) | real | Entire real class |

Approximately 50,000 images per class, balanced. Three generator families are
represented in the fake class; **the real class comes from a single source**,
which is a known weakness (see Limitations).

### Augmentation

Randomised at training time to prevent the model from keying on the dataset's
own processing pipeline:

- random JPEG re-encode, quality 30–95 (p = 0.7)
- random down-then-up rescale to 50–100% (p = 0.5)
- random resized crop (scale 0.7–1.0), horizontal flip
- colour jitter, occasional greyscale (p = 0.05), occasional blur (p = 0.2)

This exists because V1 reached 96.94% while having learned the dataset's
resize/JPEG signature rather than generator artefacts.

### Optimisation

AdamW, lr 3e-4 with cosine annealing to 1e-5, batch 128, label smoothing 0.05,
10 epochs, transfer-learned from ImageNet weights.

### Checkpoint selection

The saved checkpoint is **not** the one with the best clean accuracy. It is
selected on the mean of *robust* accuracy (JPEG q40 copies of the validation
set) and *TPDN holdout* accuracy, so the winner is the epoch that generalises
best rather than the one that fits the training distribution best.

---

## Evaluation

| Metric | Value | What it measures |
|---|---|---|
| Validation accuracy | **99.90%** | 5,000 balanced held-out images |
| Robust validation | **99.18%** | The same images re-encoded at JPEG q40 |
| TPDN holdout | **100.00%** | 1,000 thispersondoesnotexist faces excluded from training |
| Held image set | **9 / 9** | 5 StyleGAN2 fakes + an authentic portrait at 4 resolutions |
| 140k test split | *not recorded* | The export was rebuilt from a resume checkpoint, before the test cell ran |
| DFDC (face-swap) | *not evaluated* | Out of scope for V3 — see Limitations |

**Reading these honestly:**
- TPDN holdout contains only fakes, so on its own it could be gamed by a model
  that always answers "fake". Validation stays at 99.90% on a balanced set,
  which rules that out. Both numbers are needed; neither is sufficient alone.
- The nine held images are a sanity check, not a benchmark. No standard
  benchmark (FaceForensics++, Celeb-DF, DFDC) has been run, so these numbers
  cannot be compared against published results.
- **Every figure above is accuracy.** Precision, recall, F1, specificity,
  ROC-AUC, PR-AUC, false-positive rate and false-negative rate have never
  been computed for V3. Accuracy on a balanced set hides the one thing that
  decides whether a detector is usable — see below.

### The number that is missing

**V3's false-positive rate is unmeasured.**

A false positive is an authentic photograph the model calls fake. It is the
expensive error: a missed forgery leaves someone where they already were, an
accusation puts them somewhere worse. Nothing in the table above measures it,
because measuring it needs a set of genuine photographs and this repository
has none — the training real class was FFHQ, which lives on Kaggle, and no
labelled real set has ever been scored through the deployed pipeline.

Two observations that are *not* a substitute for that measurement, but are
the reason it is worth doing:

- A 2687px authentic portrait once scored **0.94 fake**, and a pristine
  camera original **0.95 fake**. Both were fixed (resolution cap, compression
  normalisation) — but they were found by hand, one image at a time.
- Validation is 99.90% on a set whose real class is FFHQ only. FFHQ is
  curated, aligned, and nothing like a photo off a phone.

`eval_data/README.md` says exactly which folders to fill and
`scripts/evaluate.py` produces the number. It is one afternoon of work and
it is the highest-value measurement left in this project.

### What has been measured with the new harness

Detection survives platform-style processing — on a very small sample:

| Condition | What was done | Detection | Mean P(fake) |
|---|---|---|---|
| original | q95 re-save | 5 / 5 | 0.975 |
| phone | 1440px cap, q92 | 5 / 5 | 0.974 |
| screenshot | 1080px cap, PNG | 5 / 5 | 0.974 |
| social | 720px cap, q60 | 5 / 5 | 0.975 |
| re-encode | q55 then q40 | 5 / 5 | 0.975 |

Same five StyleGAN2 faces through each condition. The files genuinely
differ — 277 KB, 44 KB and a 1.3 MB PNG for one image — yet the score moves
by at most **0.006**. That is the compression normalisation and the 224px
face crop doing their job: both were added to stop false positives, and they
buy robustness as a side effect.

Read it for what it is: **5 images, one generator, no real photographs**. It
shows the preprocessing is not fragile. It says nothing about the
false-positive rate, and nothing about any other generator.

### How the numbers are produced

```
scripts/ds_metrics.py     the arithmetic — one implementation
scripts/evaluate.py       runs the live engine over eval_data/, writes
                          predictions.csv, prints the metric block, the
                          per-source table, a threshold sweep and the
                          in-domain vs unseen comparison
scripts/metrics_test.py   40 known-answer tests on the arithmetic
scripts/split_test.py     24 tests that the V4 training split cannot leak
```

The training notebook computes **no** metrics. It writes raw scores to
`predictions_*.csv`, and `evaluate.py --from-csv` turns them into the table.
A Kaggle number and a local number therefore come from the same code, and
anything published here can be recomputed from the CSV that produced it.

Scoring goes through `inference.score_image`, the same face crop,
compression normalisation and flip-averaging a real upload gets. A benchmark
with its own preprocessing measures a model nobody is served by.

### Version history

| Version | Val | Robust | TPDN | What changed |
|---|---|---|---|---|
| V1 | — | — | — | 25k/class, 3 epochs, light augmentation. 96.94% test, but had learned the pipeline fingerprint |
| V2-Heavy | 99.40% | 98.54% | — | Full data, 10 epochs, anti-shortcut augmentation, robust-selected |
| **V3-Max** | **99.90%** | **99.18%** | **100.00%** | MobileNetV3-Large, three generator families |

Every version is kept in `models/archive/`; rolling back is one file copy.

---

## Preprocessing at inference

Two steps exist because of failures found in testing, not for tidiness:

1. **Resolution cap, 1024px.** A 2687px camera original scored 0.94 fake while
   the same photo at 1024px scored 0.02 — cropping out of a very large image
   produces a downsampling path unlike the ~256px training faces.
2. **JPEG q88 round-trip.** Training saw recompressed faces; a pristine
   original carries high-frequency detail the model never learned as normal.
   Applied to our model only — the optional verifiers react badly to
   re-encoding.

Then: YuNet face crop (0.35 margin), resize to 224, ImageNet normalisation,
and test-time augmentation over the image and its mirror.

---

## Explainability

**Occlusion sensitivity.** The face is divided into a 6×6 grid; each cell is
blanked in turn and the drop in the winning class's score is recorded. Cells
whose removal moves the score most are the ones the model relied on. The
result is rendered as a heatmap, and the hottest cell is matched to the
nearest YuNet landmark to produce a grounded sentence such as *"Model
attention concentrated around the eye region."*

This reports **where** the model looked, never **what** is wrong with the face.
It needs only forward passes, so it behaves identically on both backends —
Grad-CAM, used previously, requires gradients the ONNX runtime cannot provide.

---

## Limitations

1. **Face-swap deepfakes are not covered.** V3 learned fully synthesised
   faces. Face-swaps (DeepFaceLab / DFDC-style, the common video deepfake)
   leave a different artefact family — blending seams rather than generator
   fingerprints. A real DFDC video was scored 97% *real* in testing. Closing
   this is what the V4 notebook exists for.
2. **New generators are not guaranteed.** V2 scored StyleGAN2 faces at
   0.02–0.49 until StyleGAN2 was added to training. A 2027 generator may be
   invisible to it in exactly the same way. This is an open research problem,
   not a defect: the $1M DFDC winner scored 82% on its own test set and around
   65% on unseen deepfakes.
3. **The real class comes from one source (FFHQ).** Anything unlike an FFHQ
   portrait — different camera, lighting, framing — is under-represented, which
   is why the resolution and compression normalisation steps above were needed.
4. **Heavily processed media degrades detection.** Screenshots, repeated
   re-compression and platform re-encoding strip the evidence. This affects
   every detector; published measurements put the drop at 15–35%.
5. **Faces only.** Landscapes, documents and objects produce meaningless
   verdicts. The face detector's failure to find a face is not reported to the
   user; the whole frame is analysed instead.
6. **Small evaluation set.** Nine held images plus dataset splits. No
   demographic breakdown has been run, so fairness across skin tone, age and
   gender is **unmeasured** — a real gap, not a claim of fairness.
7. **Video is frame-wise.** Temporal signals (flicker, blink rate, lip-sync)
   are unused, and per-frame scores are averaged, so a partially manipulated
   clip can be diluted.

---

## Ethical notes

- Verdicts are probabilistic. The UI always shows a confidence score, and the
  report carries a disclaimer that results are not forensic evidence.
- Analysis happens on the machine running the backend; uploads are deleted
  once the verdict is ready. If the app is ever hosted publicly, the privacy
  wording must change to match — photos would then leave the user's device.
- Feedback collected from users records the verdict and a thumbs up/down only.
  It is never fed back into the model automatically; retraining is manual and
  batched, to avoid a model reinforcing its own errors.
