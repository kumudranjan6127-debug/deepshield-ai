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
| **Card updated** | 2026-08-11 |
| **See also** | [BENCHMARK.md](BENCHMARK.md) · [LIMITATIONS.md](LIMITATIONS.md) · [ARCHITECTURE.md](ARCHITECTURE.md) · [SECURITY.md](SECURITY.md) |

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
- **Every figure above is accuracy on data related to training.** The full
  metric set — precision, recall, F1, specificity, ROC-AUC, PR-AUC and the
  false-positive rate — is measured separately against an out-of-domain set
  and reported in [BENCHMARK.md](BENCHMARK.md).

### The number that was missing

**Measured 2026-08-11: 0 false positives across 501 distinct people, 95%
upper bound 0.60%** — on LFW, deliberately not FFHQ. The real and fake score
distributions do not overlap: highest real 0.1074, lowest fake 0.9689. Full
detail in [BENCHMARK.md](BENCHMARK.md).

What follows is why that measurement mattered, and what it still does not
cover: LFW is press photography, and the app receives phone photographs.

A false positive is an authentic photograph the model calls fake. It is the
expensive error: a missed forgery leaves someone where they already were, an
accusation puts them somewhere worse. Nothing in the training table measures
it, because the real class there *is* the training distribution.

`0 / 524` is not a claim that the rate is zero. With no events in 501
independent trials, the honest statement is the 95% upper bound: **below
0.60%**.

The gap that remains is specific. Both false positives ever found by hand
were large or pristine images:

- a 2687px authentic portrait scored **0.94 fake** (fixed by the 1024px cap)
- a pristine camera original scored **0.95 fake** (fixed by the JPEG round trip)

LFW would have caught neither — its images are 250×250 press photographs,
neither large nor pristine. Phone photography is still unscored.

`scripts/fetch_real_faces.py` builds the LFW set and
`scripts/evaluate.py --target-fpr 0.01` produces the number. The remaining
gap — ordinary phone photographs — is the highest-value data this project
still lacks.

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

### Calibration — what the percentage is allowed to claim

**Measured, and the shape is the problem.** ECE 0.0242, MCE 0.1074, Brier
0.0006 across 592 images. Those are good numbers for a model whose outputs
are almost all 0.02 or 0.97 — and that is exactly the caveat.

| Question | Answered by | Status |
|---|---|---|
| Does it rank fakes above reals? | ROC-AUC | **1.0000** on the measured set |
| When it says 0.9, is it right 90% of the time? | ECE, Brier, reliability | **0.0242** — but over two occupied bins |

`/api/health` still reports `"calibrated": false`. A reliability curve with
two populated bins has demonstrated nothing about the range in between, and
**591 of 592 verdicts landed in a single certainty band**.

A network trained with cross-entropy is usually over-confident. This one is
not, on this data — but this data never put it in a difficult position. That
is still the reason for the wording change:

> ~~97% probability this image is fake~~
> **Detection confidence: 97% — very strong evidence**

The verdict carries a `certainty` band alongside `confidence`:

| Band | Confidence | Reachable |
|---|---|---|
| `very_strong` | 90–100 | yes |
| `strong` | 70–90 | yes |
| `uncertain` | 30–70 | only 50–70 |
| `low_evidence` | 0–30 | **never** |

`confidence` is `max(p, 1−p)` for two classes, so it cannot fall below 50 and
the bottom band is unreachable. The cut points are **provisional** — they are
the values specified for Phase 5, not values derived from data. They live in
`backend/config.py`, are published by `/api/health` so the frontend never
holds a copy, and `scripts/evaluate.py` prints the observed accuracy and
occupancy of each band so they can be replaced with measured ones.

### Video — how frames become one verdict

The classifier is an image model. Video is handled by sampling ~1 frame per
second (60 at most), scoring each frame through the identical pipeline, and
combining the results:

| Summary | Weight | What it is blind to without the others |
|---|---|---|
| median | 0.40 | manipulation confined to a few seconds |
| mean | 0.25 | nothing much; it is the level, and it dilutes |
| top-k mean, k = 15% | 0.35 | it over-reacts to a run of bad frames |

Two failure modes this is shaped around, both real:

- **Averaging dilutes.** A face-swap that only holds while the subject faces
  the camera can be 20 frames of 0.95 inside 60 frames of 0.05. The mean is
  0.35 and the clip passes.
- **Maximum accuses.** One motion-blurred frame scoring 0.97 would, under a
  max, call an authentic video a deepfake. Top-k with k > 1 needs the
  evidence to persist.

The weights **have never been fitted**. No labelled video set has been
scored, so they are reasoned rather than measured, and they lean on the
median because a false accusation costs more than a missed forgery. That
lean is a real trade: a clip with a third of its frames strongly flagged
still comes out "real", which is why the response reports
`suspiciousFrames` and the timestamps whether or not the verdict says fake.
Every component is returned, so the combination can be recomputed — or
replaced — from the response alone.

`tests/test_video.py` pins the behaviour against sequences whose answer is
obvious by construction, including the one that matters: 59 calm frames and
one disaster must stay "real".

### Temporal signals — reported, never counted

Four cheap consistency measures come out of data the frames already
produced (face box, five landmarks, a 32×32 thumbnail) — face position
jitter, face size jitter, landmark jitter, and appearance continuity
between consecutive crops.

**None of them touches the verdict.** There is no evidence for what value of
"landmark jitter" means manipulation, and a signal nobody has validated must
not be allowed to change an answer. They are shown so a person can look, and
so that when labelled video does arrive there is already something to
correlate against.

Measured cost: **1.1 s for 12 frames** on the target CPU, no extra forward
passes and no extra face detection. That is why this is four descriptive
numbers and not a video transformer.

### How the numbers are produced

```
scripts/ds_metrics.py     the arithmetic — one implementation
scripts/evaluate.py       runs the live engine over eval_data/, writes
                          predictions.csv, prints the metric block, the
                          per-source table, a threshold sweep and the
                          in-domain vs unseen comparison
tests/test_metrics.py   40 known-answer tests on the arithmetic
tests/test_split.py     24 tests that the V4 training split cannot leak
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
nearest YuNet landmark to produce a grounded sentence: *"Prediction was most
sensitive to the eye region."*

**That sentence used to read "Model attention concentrated around the eye
region", and it was wrong.** Occlusion sensitivity does not observe
attention. It hides a patch and measures how far the output moves — a
statement about the *prediction*, not about the network's internals. The two
can disagree: a region the model never attends to can still swing the score
because hiding it changes the image statistics. The UI called the heatmap
"Grad-CAM" as well, which it has never been. Both are corrected.

The response ranks **every** region the prediction leaned on, not only the
strongest: `[{"name": "the eye region", "weight": 1.0}, {"name": "the mouth
area", "weight": 0.44}]`. Each weight is the largest normalised score drop
that region produced, so the bullets a user reads are ranked measurements
rather than a narrative — and one region is a poor summary when a face gives
itself away in two places.

So this reports **which regions the prediction depended on**, never **what is
wrong with the face**. It needs only forward passes, so it behaves identically
on both backends — Grad-CAM requires gradients the ONNX runtime cannot
provide.

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
7. **Video is still frame-wise.** The classifier sees single frames and
   nothing else; it cannot see flicker, blink rate or lip-sync desync,
   which is where most video deepfakes actually give themselves away.
   Frames are no longer merely averaged — median, mean and a top-k mean are
   combined, so a clip manipulated for only part of its length is no longer
   diluted away — but the combination weights have never been fitted
   against labelled video, and four face-consistency signals are computed
   and shown without being allowed to affect the verdict, for the same
   reason. See `KNOWN_ISSUES.md`.

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
