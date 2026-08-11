# V4 Dataset Audit and Training Plan

**Status: audit only. Nothing trained, nothing downloaded, no production code
touched.**

Read §3 first. The most important finding is not about datasets at all — it
is that the training pipeline and the production pipeline preprocess images
differently, which means every validation number V3 ever produced describes a
model that is not the one being served.

---

## 1. Current training pipeline summary

Read from `training/DeepShield_V4_Universal.ipynb` and cross-checked against
`backend/inference.py` and `models/deepshield.onnx.json`.

| Setting | Value |
|---|---|
| Backbone | MobileNetV3-Large, ImageNet weights, classifier head → 2 classes |
| Input | 224 × 224 RGB |
| Normalisation | ImageNet — mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]` |
| Classes | `['fake', 'real']` — **index 0 = fake**, order is load-bearing |
| Optimiser | AdamW, lr 3e-4, cosine → 1e-5 |
| Loss | CrossEntropy, label smoothing 0.05 |
| Batch / epochs | 128 / 10 |
| Seed | 42 |
| Per class | 50,000 train, 2,500 validation |
| Checkpoint selection | mean of robust-val and TPDN-holdout accuracy — **not** best clean accuracy |

### Transforms as they stand

```python
train_tf = RandomRescale(p=0.5, lo=0.5)          # down-then-up, kills resolution fingerprints
           RandomJPEG(p=0.7, quality=30..95)     # kills compression fingerprints
           RandomResizedCrop(224, scale=0.7..1.0)
           RandomHorizontalFlip()
           ColorJitter(0.15, 0.15, 0.1)
           RandomGrayscale(p=0.05)
           RandomApply(GaussianBlur(3), p=0.2)
           ToTensor(); Normalize(MEAN, STD)

eval_tf  = Resize((224, 224)); ToTensor(); Normalize(MEAN, STD)
robust_tf= FixedJPEG(40); Resize((224,224)); ToTensor(); Normalize(MEAN, STD)
```

`RandomRescale` and `RandomJPEG` exist because V1 reached 96.94% having
learned the dataset's own resize/JPEG signature rather than generator
artefacts. That lesson is sound and must be kept.

---

## 2. Current dataset summary

| Source | Class | Role | Notes |
|---|---|---|---|
| 140k Real and Fake Faces (`xhlulu`) | both | base | StyleGAN1 fakes; real split is FFHQ |
| thispersondoesnotexist (`almightyj`) | fake | StyleGAN2 | 1,000 held out |
| Fakefaces (`hyperclaw79`) | fake | StyleGAN2 hi-res | |
| Stable Diffusion faces (`mohannadaymansalah`) | fake | diffusion | currently `UNSEEN_FAMILY` — withheld |
| DFDC face crops (`dagnelies/deepfake-faces`) | both | face-swap | 1,000 held out, **split by identity group** |

**Real class comes from two sources: FFHQ and DFDC real frames.** FFHQ is
curated, aligned, studio-ish. That narrowness is the reason the
false-positive question stayed open so long.

The identity-group split for DFDC is already correct and already tested
(`tests/test_split.py`): each fake belongs to the `original` real video it
was made from, and whole identities move together. On synthetic data with the
same structure, the previous file-level split leaked **78 of 100** held-out
files.

---

## 3. 🔴 Train/serve skew — the finding that matters most

Production (`backend/inference.py`) and training preprocess differently. This
is not a dataset problem and no amount of new data fixes it.

| Step | Production | Training | Consequence |
|---|---|---|---|
| Resolution cap | 1024px longest side, `INTER_AREA` | none | Large images take a path training never saw |
| **Face crop** | **YuNet detection, largest face, 0.35 margin** | **none — datasets are pre-cropped**, then `RandomResizedCrop` | **The crop geometry served is not the crop geometry trained** |
| Compression | **JPEG q88 on every input, unconditionally** | `RandomJPEG(30..95)` at p=0.7 — 30% get none; `eval_tf` gets none | Validation measures a pipeline nobody runs |
| Resize to 224 | PIL BILINEAR, after crop | `Resize`/`RandomResizedCrop` on the whole image | |
| Video | 60 frames sampled at 1 fps, each through the full path | **one pre-extracted crop per DFDC video** | The model never trains on the frame distribution it is served |

`YuNet`, `FaceDetector`, `face_crop` and `detect` are all **absent from the
training notebook entirely.**

### Why this is serious

The two hand-found false positives were both caused by exactly this class of
mismatch — a 2687px portrait scored 0.94 fake because of its downsampling
path, and a pristine camera original scored 0.95 because of its compression
domain. Both were patched *in production* (the 1024 cap, the q88 round trip)
rather than trained for. Those patches work, but they are compensating for a
model that was trained on a different distribution.

**This must be fixed before V4 is trained, or V4 inherits the same problem.**

### The fix

Preprocess the training corpus with the *same code path* production uses:
extract faces with YuNet at 0.35 margin, apply the 1024 cap, and store the
crops. Then `eval_tf` must include the q88 round trip so validation measures
the served pipeline. See §C.

---

## 4. Recommended datasets and exact purpose

> Every dataset below requires a licence acceptance or a request form.
> **None should be downloaded automatically.** Sizes and terms stated here
> must be re-verified at the source before use — they change.

### Train

| Dataset | Purpose | Why this one |
|---|---|---|
| **FaceForensics++** | Primary face-swap and reenactment training | The field's reference set. Four manipulation methods (Deepfakes, Face2Face, FaceSwap, NeuralTextures) at three compression levels (raw / c23 / c40). The compression levels alone are worth more than any augmentation we could invent, because they are real H.264 artefacts, not simulated JPEG |
| **DFDC** | Scale and actor diversity | Largest public set, ~960 paid consenting actors, deliberately varied lighting, pose, background and ethnicity. Already partly integrated |
| **Existing generated-face sets** (SG1, SG2, TPDN, diffusion) | Fully-synthetic faces | The family V3 already handles; must not regress |
| **A modern phone-photograph real set** | Close the FPR gap | **The single most valuable missing input.** See §4b |

### Test only — never train, never tune

| Dataset | Purpose | Why held out |
|---|---|---|
| **Celeb-DF v2** | The honest generalisation number | Higher visual quality than FF++; detectors that score well on FF++ have historically dropped sharply here. If V4 does well on Celeb-DF without ever seeing it, that is a real result |
| **DeeperForensics-1.0** | Robustness under real-world distortion | Ships a deliberate perturbation taxonomy (compression, blur, noise, colour, contrast, at graded severities). Use its *test* protocol as a robustness curve |
| **One withheld FF++ method** | Cross-manipulation generalisation | Train on three, test on the fourth. Rotate which one across runs |

### 4b. The dataset I would add, and why

**A few hundred ordinary phone photographs of real people.**

Not a public dataset — collected or contributed. Justification:

- The app's actual traffic is phone photographs. Nothing in training or
  evaluation currently represents them.
- LFW gave 0 false positives across 501 people, but LFW is 250×250 press
  photography with 2000s web compression. It contains nothing large and
  nothing pristine, and **both false positives ever observed were exactly
  those two kinds of image**.
- It costs nothing to obtain and it directly addresses the highest-severity
  open question.

Two public stand-ins if a private set is not possible — both must be checked
for face-image licensing before use:

- **Open Images / COCO person crops** — genuine uncurated photography, wide
  camera and quality range, permissive licensing on the annotations
- **CelebA-HQ** — a *different* curated set from FFHQ; useful as a second
  curated source, but it does not solve the phone-photo gap

### Deliberately not recommended

- **FFHQ as an evaluation set** — it is the training real class. Scoring
  against it measures memorisation of the training distribution.
- **Any generator whose images appear in training** as a "cross-dataset"
  test. That is not cross-dataset.
- **Scraped social media content** — consent and licensing are unresolvable,
  and re-encoding has already destroyed the artefacts.

---

## 5. Licensing and provenance notes

**Verify each at source before downloading. Do not assume these terms are
current.**

| Dataset | Access | Broad terms | Redistributable? |
|---|---|---|---|
| FaceForensics++ | Signed EULA / request form to the authors (TU Munich) | Research use; derived from public YouTube video | **No** |
| Celeb-DF v2 | Request form to the authors | Research use only; celebrity footage from YouTube | **No** |
| DFDC | Licence acceptance (Meta / Kaggle) | Research use; **actors gave explicit consent** — the cleanest provenance of the four | **No** |
| DeeperForensics-1.0 | Request form / GitHub agreement | Research use; 100 paid consenting actors | **No** |
| FFHQ | Creative Commons / public-domain Flickr images, per-image licences | Non-commercial research generally accepted | Per-image |
| LFW | Public, academic | Unrestricted for research | Yes |
| Generated-face sets on Kaggle | Per-dataset licence | Varies — **check each** | Varies |

Practical consequences:

1. **Nothing goes in the git repository.** `eval_data/` and any new
   `datasets/` directory stay gitignored, as today.
2. **The manifest, not the media, is the artefact.** A CSV of file hashes,
   labels and group ids is redistributable and reproducible; the images are
   not.
3. **Consent matters and differs.** DFDC and DeeperForensics used paid
   consenting actors. FF++ and Celeb-DF are built from public figures'
   YouTube footage without individual consent. That is a fact worth recording
   in the model card, not a reason to avoid them for research.
4. **A model trained on these datasets inherits their restrictions.**
   Publishing weights trained on FF++ or Celeb-DF may not be permitted.
   Resolve this *before* training, not after.

---

## 6. Recommended train / validation / test strategy

Three tiers, and the boundaries never move.

```
TRAIN            model sees it, gradients flow
  FF++ (3 of 4 methods, all compressions)
  DFDC (identity groups A)
  SG1 / SG2 / TPDN / diffusion
  real: FFHQ + DFDC real (groups A) + phone photos (subset)

VALIDATION       checkpoint selection and early stopping only
  same sources, disjoint identity groups
  + a robust copy (JPEG q40) — the selection metric, as today

TEST — SEALED    scored once, at the end, never tuned against
  Celeb-DF v2                     (never trained on, any part)
  DeeperForensics-1.0 perturbation ladder
  FF++ withheld method            (cross-manipulation)
  DFDC identity groups B          (cross-identity)
  LFW + phone photos              (false-positive rate)
```

Rules that make this valid:

- **The sealed test set is scored once per model version.** If a number from
  it changes a hyperparameter, it is no longer a test set — it is validation,
  and it must be relabelled as such in the report.
- **Checkpoint selection uses validation only**, on the existing robust +
  holdout mean. The `UNSEEN_FAMILY` must never enter checkpoint selection —
  that guard is already in the notebook and must survive.
- **Validation accuracy is never reported as generalisation.** It is reported
  as what it is: the number used to pick an epoch.

---

## 7. Identity and source leakage prevention

The failure mode, concretely: FF++ builds several manipulations from the same
source YouTube video. DFDC builds several fakes from one original. Celeb-DF
has many clips per celebrity. Split any of those by *file* and the model is
tested on a face it memorised.

**Group key by dataset:**

| Dataset | Group key |
|---|---|
| FF++ | source video id — and **both** videos of a manipulated pair share it |
| DFDC | `original` column from `metadata.csv` (already implemented) |
| Celeb-DF | celebrity id — sealed anyway, but the rule still applies |
| DeeperForensics | actor id |
| FFHQ / TPDN / SG2 / diffusion | one image = one group (independent draws) |
| LFW | person name from filename (already implemented) |
| Phone photos | photographer / subject, whichever is coarser |

**Enforcement, not intention:**

1. Every manifest row carries a `group` column. No row may be split from its
   group.
2. Splitting operates on the **set of groups**, never on the set of files.
3. A hard assertion after splitting: `train_groups ∩ val_groups ∩ test_groups
   = ∅`, failing loudly. This already exists in the V4 notebook and must be
   extended to every new source.
4. `tests/test_split.py` lifts the split code out of the notebook and runs it
   on synthetic data whose leakage structure is known — including a check
   that the *old* file-level split really did leak, so the fix cannot decay
   into a no-op. Extend this test as sources are added.

---

## 8. Duplicate detection strategy

None exists today. Required, because FF++ and DFDC both derive from YouTube
and near-duplicate frames are guaranteed within any video.

Three layers, cheapest first:

| Layer | Catches | Method |
|---|---|---|
| Exact | Byte-identical files across sources | SHA-256 of file bytes |
| Near-duplicate frames | Consecutive frames of one clip | Perceptual hash (dHash/pHash, 64-bit) — drop within Hamming ≤ 4 of a kept frame **inside the same group** |
| Cross-split identity | The same face in two splits | Group key (§7) — the real defence |

Perceptual hashing must run **within a group, not globally**: two different
people can hash close, and dropping them loses real diversity. The group key
is what prevents cross-split contamination; pHash is for reducing redundancy
inside a group so 300 near-identical frames do not count as 300 samples.

Record every drop with its reason in the manifest. A silently shrinking
dataset is a debugging nightmare later.

---

## 9. Class balancing strategy

**Balance by group, not by file.** Balancing 50,000 fakes against 50,000
reals means nothing if the fakes come from 200 identities and the reals from
50,000.

| Axis | Target |
|---|---|
| fake vs real | 1 : 1 by **image count**, and no worse than 1 : 2 by **group count** |
| Within fake | face-swap ≈ 50%, fully-generated ≈ 50% — a deliberate change from V3's 35/65, because face-swap is the known blind spot |
| Within face-swap | FF++ methods roughly equal; no single method above 40% |
| Within real | no single source above 50% — FFHQ currently dominates and that is the FPR risk |
| Per identity | cap contributions, e.g. ≤ 40 crops per group, so no actor dominates |

Do **not** use class weights to paper over imbalance. Sample to the target
and record the achieved composition in the manifest; a printed composition
table is how the next person sees what the model actually saw.

---

## 10. Frame sampling strategy for videos

Today: **one pre-extracted crop per DFDC video.** Production samples 60
frames at 1 fps. That gap should close from both ends.

Recommended for training:

- **Uniform temporal sampling**, not the first N frames — the beginning of a
  clip is not representative of it.
- **8–16 frames per video** for training. Beyond that, returns fall off
  quickly and the identity cap (§9) binds first.
- **Every frame inherits its video's group.** Non-negotiable.
- **Deduplicate within the video** with pHash (§8) — a static shot yields
  near-identical frames.
- **Keep the frame index and timestamp in the manifest**, so a suspicious
  result can be traced back to the exact frame.

For evaluation, mirror production exactly: 1 fps, capped at 60 frames, then
aggregate with the same median/mean/top-k weights the app uses. Anything else
measures a different system.

---

## 11. Face detection and cropping strategy

**Use YuNet, at 0.35 margin, through the same code production runs.** This is
the §3 fix and it is the highest-value change in this plan.

```
frame or image
   ↓ cap at 1024px longest side (INTER_AREA)      ← matches production
   ↓ YuNet detect, score ≥ 0.6
   ↓ largest face
   ↓ expand box by 0.35 × max(w, h), clamp to frame
   ↓ save crop as PNG (lossless) + record box, landmarks, detection score
```

Details that matter:

- **Save crops losslessly.** Compression is applied later as augmentation; a
  lossy intermediate bakes a JPEG generation into every sample and is exactly
  the shortcut V1 learned.
- **Record the detection score and box** in the manifest. It makes §11's
  quality filter possible and makes failures auditable.
- **No face found → the sample is dropped from training**, and the drop is
  recorded. Production analyses the whole frame in that case, but training on
  frames with no face teaches the model to classify backgrounds.
- **One face per image.** Where multiple faces appear, the largest is the
  subject — same rule as production. Record how many were found.

---

## 12. Image quality filtering strategy

Filter, and record what each filter removed. Silent filtering is how a
dataset quietly becomes something other than what its description says.

| Filter | Threshold | Rationale |
|---|---|---|
| Detection confidence | drop < 0.6 | A doubtful box is a doubtful label |
| Face size | drop crops whose detected box is < 64 px on its short side | Below that there is no artefact left to detect |
| Blur | variance of Laplacian below a per-dataset percentile (e.g. bottom 5%) | Motion blur destroys the signal; keep *some* so the model sees it |
| Extreme pose | landmark-derived yaw beyond ~60° | Profile faces carry almost no swap artefact |
| Over/under exposure | mean luma outside ~[15, 240] | |
| Aspect ratio | drop boxes further than 1:2 from square after margin | Usually a detector error |

**Do not filter aggressively.** A training set of only clean, frontal, sharp
faces produces a model that fails on everything else — the model is served
photographs from phones, not a casting call. Filter for *label validity*
(is there really a face here?), not for *aesthetics*.

---

## 13. Recommended augmentations

Keep what V3 has — it was derived from a real failure — and add the
production-matching step.

```python
train_tf = RandomRescale(p=0.5, lo=0.5)              # keep
           RandomJPEG(p=0.9, quality=(30, 95))       # p 0.7 -> 0.9
           JPEGQuality(88) applied at p=1.0 last     # NEW: match production
           RandomResizedCrop(224, scale=(0.8, 1.0))  # 0.7 -> 0.8, crops are already tight
           RandomHorizontalFlip()
           ColorJitter(0.15, 0.15, 0.1)
           RandomGrayscale(p=0.05)
           RandomApply(GaussianBlur(3), p=0.2)
           ToTensor(); Normalize(MEAN, STD)

eval_tf  = JPEGQuality(88); Resize((224,224)); ToTensor(); Normalize(...)
                ↑ NEW — validation must measure the served pipeline
```

Additions worth testing, each individually:

| Augmentation | Why | Risk |
|---|---|---|
| **H.264 re-encode** at CRF 23/40 | The real artefact FF++ ships; JPEG is a poor proxy for video | Slow; needs ffmpeg in the pipeline |
| **Downscale-upscale to 112px** | Simulates a small face in a large frame | Already partly covered by RandomRescale |
| Cutout / random erasing (p ≈ 0.25) | Stops reliance on one region — the occlusion map shows heavy mouth/eye dependence | Can destroy the artefact entirely at large sizes |

Explicitly **not** recommended: MixUp and CutMix. Blending a real and a fake
face produces an image whose true label is undefined, and this task's decision
boundary is exactly what that would blur.

---

## 14. Recommended dataset sizes for a practical Kaggle GPU run

Constraints: Kaggle session ≤ 12 h, ~30 GPU-hours/week, ~73 GB disk, P100 or
T4×2. MobileNetV3-Large at 224px, batch 128.

| Split | Images | Groups (min) | Notes |
|---|---|---|---|
| Train — fake | 60,000 | ≥ 3,000 | ~30k face-swap, ~30k generated |
| Train — real | 60,000 | ≥ 8,000 | FFHQ ≤ 50%, rest DFDC real + phone photos |
| Validation | 6,000 | ≥ 600 | balanced; disjoint groups |
| Sealed test | 15,000–20,000 | ≥ 1,500 | Celeb-DF + DeeperForensics + withheld FF++ method + LFW |

**120k training images is the sweet spot, not the ceiling.** At 224px with
this backbone, an epoch is roughly 10–15 minutes on a P100; 10–12 epochs fits
one session with room for evaluation. Going to 200k mostly buys near-
duplicates unless group count grows with it — and group count, not image
count, is what generalisation follows.

Preparation happens in a **separate session** from training: extract crops
once, save the manifest and a packed archive, attach it as a Kaggle dataset.
Re-extracting faces inside the training session wastes GPU hours on CPU work.

Use *Save Version → Save & Run All (Commit)* for the real run. The previous
V4 attempt was lost because an interactive session ended (`KNOWN_ISSUES` #12).

---

## 15. Completely isolated unseen test set strategy

The test set is valid only if every one of these holds:

1. **Sealed before training starts.** Group ids written to
   `manifests/test_groups.txt` and committed. Not chosen afterwards.
2. **No identity appears in train or validation.** Asserted, not intended.
3. **At least one entire manipulation method never seen** — a withheld FF++
   method.
4. **At least one entire dataset never seen** — Celeb-DF v2, in full.
5. **Scored once**, after checkpoint selection is complete.
6. **No hyperparameter, threshold or augmentation is chosen using it.** If
   one is, the set is burned and must be relabelled validation.
7. **Scored through the production path** — `inference.score_image`, so the
   number describes the served system.

A practical safeguard: keep the sealed test manifest in a separate file that
the training notebook never reads. It should be physically awkward to cheat.

---

## 16. Metrics that must be reported

Accuracy alone is not reportable. For every evaluation set:

**Detection**
`accuracy · precision · recall · specificity · F1 · ROC-AUC · PR-AUC`

**Errors, separately and prominently**
`FPR (real called fake)` · `FNR (fake called real)` — with the **95%
confidence interval**, and the rule-of-three upper bound when the count is
zero.

**Calibration**
`ECE · MCE · Brier` plus a reliability diagram, and **bin occupancy** — V3's
ECE of 0.024 looks excellent until you see that 591 of 592 predictions fell
in one bin.

**Sample-size honesty**
`n images` and `n independent groups`, always together. The group count is
the real sample size.

**Generalisation — the headline**

| Reported as | Meaning |
|---|---|
| In-domain accuracy | manipulation methods seen in training |
| Cross-manipulation accuracy | the withheld FF++ method |
| **Cross-dataset accuracy** | **Celeb-DF, never seen** |
| Generalisation gap | in-domain minus cross-dataset, in points |

**Robustness**
Accuracy across the DeeperForensics perturbation ladder, as a curve.

**Operating point**
Threshold and recall at a 1% false-positive budget — not just the 0.5 default.

**Cost**
Latency per image and per video second, peak RSS, on CPU. A model that
generalises but does not fit the 512 MB tier has not shipped.

All of these already exist in `scripts/ds_metrics.py` and print from
`scripts/evaluate.py`. **The training notebook must not compute metrics.** It
writes `predictions.csv`; the repo computes the table. One implementation of
the arithmetic, so a Kaggle figure and a local figure cannot disagree.

---

# A. Proposed directory structure

```
datasets/                          gitignored, never committed
  raw/                             as downloaded, untouched
    faceforensics/
      original_sequences/
      manipulated_sequences/{Deepfakes,Face2Face,FaceSwap,NeuralTextures}/{raw,c23,c40}/
    dfdc/
    celebdf_v2/                    SEALED — extraction only, never into train
    deeperforensics/               SEALED
    generated/{sg1,sg2,tpdn,diffusion}/
    real_phone/                    the gap; contributed photographs

  crops/                           YuNet output, lossless PNG, production geometry
    <dataset>/<group_id>/<frame_id>.png

  manifests/
    all.csv                        every crop, one row each
    train.csv  val.csv  test.csv   derived, group-disjoint
    test_groups.txt                SEALED — written once, committed
    dropped.csv                    every rejected crop, with the reason
    composition.md                 achieved balance, generated

  packed/
    v4_train.tar                   what actually uploads to Kaggle
    v4_manifest.csv

scripts/
  prepare/                         NEW — none of this touches production
    01_index_raw.py                walk raw/, emit provenance rows
    02_extract_faces.py            YuNet crops via the production path
    03_dedupe.py                   sha256 + pHash within group
    04_quality_filter.py           §12, writes dropped.csv
    05_build_splits.py             group-disjoint, asserts disjointness
    06_pack.py                     tar + manifest for upload
```

`datasets/` is added to `.gitignore`. The **manifests are the artefact worth
keeping** — they are small, redistributable, and fully describe the run.

---

# B. Dataset manifest format

One CSV, one row per crop. Every downstream step reads this and nothing else.

```csv
crop_path,dataset,source_file,group,label,manipulation,compression,
frame_index,timestamp,face_score,face_box,n_faces,width,height,
sha256,phash,split,drop_reason
```

| Column | Meaning |
|---|---|
| `crop_path` | relative to `datasets/crops/` |
| `dataset` | `ff++`, `dfdc`, `celebdf`, `deeperforensics`, `ffhq`, `tpdn`, `sg2`, `diffusion`, `lfw`, `phone` |
| `source_file` | the original video or image — provenance |
| **`group`** | **identity key. Nothing splits across this** |
| `label` | `fake` or `real` |
| `manipulation` | `deepfakes`, `face2face`, `faceswap`, `neuraltextures`, `stylegan2`, `diffusion`, `none` |
| `compression` | `raw`, `c23`, `c40`, `jpeg`, `unknown` |
| `frame_index`, `timestamp` | video only; traceability |
| `face_score`, `face_box`, `n_faces` | YuNet output; makes filtering auditable |
| `sha256`, `phash` | duplicate detection |
| `split` | `train`, `val`, `test`, `dropped` |
| `drop_reason` | empty unless dropped — `no_face`, `low_score`, `too_small`, `blurry`, `duplicate`, `extreme_pose` |

Rules:

- **Rows are never deleted.** A dropped crop keeps its row with a reason, so
  the corpus is fully reconstructible and shrinkage is visible.
- `group` is namespaced by dataset (`ff++:0042`), so ids cannot collide.
- The manifest is committed; the crops are not.

---

# C. Preprocessing pipeline

**Design rule: the crop that trains and the crop that is served come out of
the same function.**

```
raw video / image
   │
   ├─ video: uniform sample 8–16 frames, keep index + timestamp
   │
   ▼
cap at 1024px longest side (INTER_AREA)          ── production parity
   ▼
YuNet detect (score ≥ 0.6), largest face
   ├─ none found → drop, reason=no_face
   ▼
expand box 0.35 × max(w,h), clamp to frame
   ▼
save PNG (lossless) + record box, score, landmarks
   ▼
sha256 + pHash → dedupe within group
   ▼
quality filters (§12) → dropped.csv
   ▼
manifest row
   ▼
────────────── training time ──────────────
   ├─ train_tf:  RandomRescale → RandomJPEG(30..95, p=0.9)
   │             → JPEG q88 → RandomResizedCrop(224, 0.8..1.0)
   │             → flip / jitter / grayscale / blur → normalise
   └─ eval_tf:   JPEG q88 → Resize(224) → normalise
                     ↑ the step that makes validation mean something
```

The extraction stage (`02_extract_faces.py`) should **import the production
crop function** rather than reimplement it. If that is impractical inside a
Kaggle session, the function must be copied verbatim with a comment naming
the source, and a test must assert the two produce identical crops on the
sample images.

---

# D. Exact next steps before training

Ordered. Nothing here trains anything.

| # | Step | Output | Blocked on |
|---|---|---|---|
| 1 | **Request dataset access** — FF++, Celeb-DF v2, DeeperForensics-1.0. Accept the DFDC licence | Approval emails | Days to weeks — **start today**, it is the long pole |
| 2 | **Resolve the weight-publishing question** — may a model trained on FF++/Celeb-DF be published? | A written answer in `MODEL_CARD.md` | Reading the EULAs |
| 3 | **Collect phone photographs** — a few hundred, real people, ordinary phones, with permission | `datasets/raw/real_phone/` | Nobody but you |
| 4 | **Write `scripts/prepare/`** — the six steps in §A. Test each on the sample faces already in the repo | Working pipeline, no datasets needed yet | Nothing |
| 5 | **Fix the train/serve skew (§3)** — extraction via the production crop path; `eval_tf` gains the q88 round trip | A test asserting training and production crops match | Step 4 |
| 6 | **Extend `tests/test_split.py`** to every new source's group key | Leakage assertions that fail loudly | Step 4 |
| 7 | **Seal the test set** — write and commit `test_groups.txt` **before** any training | A committed file | Steps 1, 4 |
| 8 | **Dry run on 5% of the data** — full pipeline end to end, verify composition and leakage assertions | `composition.md`, all assertions green | Steps 4–7 |
| 9 | **Re-baseline V3** on the sealed test set through `scripts/evaluate.py` | V3's honest cross-dataset numbers | Steps 1, 7 |

Step 9 is worth pausing on: **V3 must be scored on the sealed test set before
V4 is trained.** Without that, there is no baseline to compare against, and
"V4 is better" becomes an assertion instead of a measurement.

---

## §15b. V3 vs V4 comparison protocol

*(Requested as item 15; recorded here so it sits next to the steps that
produce it.)*

Both models scored by the **same code, on the same sealed set, on the same
day**, through `scripts/evaluate.py --from-csv`.

| Rule | Why |
|---|---|
| Same test manifest, unchanged | Otherwise the comparison measures the data |
| Same preprocessing — the production path | Otherwise it measures the pipeline |
| Same threshold (0.5) **and** the 1%-FPR operating point | A model can win at one and lose at the other |
| Report per-slice, not just overall | V4 may improve face-swap and regress on generated faces; an average would hide it |
| Report latency and peak RSS | A win that does not fit the deployment is not a win |

Report as a difference table, per slice: in-domain, cross-manipulation,
cross-dataset (Celeb-DF), robustness ladder, and false-positive rate.

**V4 replaces V3 only if:**

1. Cross-dataset (Celeb-DF) accuracy improves **and**
2. False-positive rate does not get worse — measured with its confidence
   interval, not the point estimate **and**
3. Fully-generated-face detection does not regress below V3 **and**
4. Latency and memory stay within budget.

Failing any of these, V3 stays in production and V4 is archived with its
numbers. A model that detects face-swaps but starts accusing real
photographs is a worse product than one that misses face-swaps.

---

## A closing constraint

Nothing in this plan will produce a detector that catches every deepfake. It
cannot: the generators move, and a model trained today has never seen what
ships next month. What it can produce is a detector whose **failure modes are
measured and stated** — which is the only honest thing a project this size
can offer.
