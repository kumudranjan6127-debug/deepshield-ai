# V4 Dataset Provenance and Access Status

**Nothing in this document is a legal opinion.** Terms are summarised from
each dataset's public page as of 2026-08-11 and **must be re-read at the
source before use** — they change, and a summary is not a licence.

Where a licence does not clearly grant something, this document says
`UNCLEAR`. It does not say "probably fine".

---

## Status at a glance

| Dataset | Role | Access status | On this machine |
|---|---|---|---|
| FaceForensics++ | TRAIN | `NOT_REQUESTED` | `NOT_DOWNLOADED` |
| DFDC | TRAIN | `NOT_REQUESTED` | `NOT_DOWNLOADED` |
| Celeb-DF v2 | **SEALED** | `NOT_REQUESTED` | `NOT_DOWNLOADED` |
| DeeperForensics-1.0 | **SEALED** | `NOT_REQUESTED` | `NOT_DOWNLOADED` |
| Phone photographs | TRAIN + **SEALED** split | `NOT_COLLECTED` | `NOT_DOWNLOADED` |
| LFW | evaluation (FPR probe) | `ACCESSIBLE` | **`DOWNLOADED`** — 500 images |
| Generated faces (SG2/TPDN) | TRAIN | `ACCESSIBLE` | **`DOWNLOADED`** — 5 stills |
| Local test clips | pipeline exercise | `ACCESSIBLE` | **`DOWNLOADED`** — 2 clips |

**No approval has been requested yet for any of the four main datasets.**
That is the single largest thing standing between this pipeline and a real
V4. Approval is measured in days to weeks and nothing else on the critical
path takes that long.

---

## 1. FaceForensics++

| | |
|---|---|
| **Role** | Primary face-swap and reenactment training data |
| **Source** | <https://github.com/ondyari/FaceForensics> — Technical University of Munich |
| **Access** | Request form / signed EULA to the authors; a download script is issued afterwards |
| **Status** | `NOT_REQUESTED` · `NOT_DOWNLOADED` |
| **Permitted use** | Academic research, per the EULA |
| **Redistribution of the data** | **PROHIBITED** |
| **Redistribution of trained weights** | **UNCLEAR** — see below |
| **Provenance of the content** | Public YouTube video of real people, **without individual consent** |

Why it is wanted: four manipulation methods (Deepfakes, Face2Face, FaceSwap,
NeuralTextures) at three compression levels (raw, c23, c40). The compression
levels are real H.264 artefacts, which no JPEG augmentation can imitate.

`NeuralTextures` is the **withheld cross-manipulation holdout** (see
`docs/v4_sealed_groups.txt` and `scripts/prepare/sealed.py`). It must never
appear in training.

---

## 2. DFDC — Deepfake Detection Challenge

| | |
|---|---|
| **Role** | Scale and actor diversity |
| **Source** | <https://ai.meta.com/datasets/dfdc/> · also on Kaggle |
| **Access** | Licence acceptance (no individual approval wait on Kaggle) |
| **Status** | `NOT_REQUESTED` · `NOT_DOWNLOADED` |
| **Permitted use** | Research, per the DFDC licence |
| **Redistribution of the data** | **PROHIBITED** |
| **Redistribution of trained weights** | **UNCLEAR** |
| **Provenance of the content** | ~960 **paid actors who gave explicit consent** |

The cleanest provenance of the four, and the reason it is worth the disk
space. Roughly 470 GB in full; the preview subset is far smaller and may be
enough for a first run.

Grouping comes from `metadata.json`, which names each fake's `original`
clip — already implemented in `scripts/prepare/datasets.py`.

---

## 3. Celeb-DF v2 — 🔒 SEALED

| | |
|---|---|
| **Role** | **Cross-dataset generalisation. Never trained on, any part.** |
| **Source** | <https://github.com/yuezunli/celeb-deepfakeforensics> |
| **Access** | Request form to the authors |
| **Status** | `NOT_REQUESTED` · `NOT_DOWNLOADED` |
| **Permitted use** | Academic research only |
| **Redistribution of the data** | **PROHIBITED** |
| **Redistribution of trained weights** | not applicable — nothing trains on it |
| **Provenance of the content** | Celebrity YouTube footage, **without individual consent** |

This is the honest generalisation number. Detectors that score well on FF++
have historically dropped sharply on Celeb-DF, and a good score here without
ever having seen it is the only result in this plan that would mean much.

Sealed by dataset name **and** by group id, so renaming the folder does not
defeat it.

---

## 4. DeeperForensics-1.0 — 🔒 SEALED

| | |
|---|---|
| **Role** | **Robustness ladder. Never trained on.** |
| **Source** | <https://github.com/EndlessSora/DeeperForensics-1.0> |
| **Access** | Request form / agreement |
| **Status** | `NOT_REQUESTED` · `NOT_DOWNLOADED` |
| **Permitted use** | Academic research |
| **Redistribution of the data** | **PROHIBITED** |
| **Provenance of the content** | 100 **paid actors who gave consent** |

Ships a deliberate perturbation taxonomy — compression, blur, noise, colour
and contrast at graded severities. Its value is the *protocol* as much as the
data: it is a robustness curve rather than a single number.

---

## 5. Phone photographs

| | |
|---|---|
| **Role** | Close the false-positive gap. Split: most to TRAIN, a sealed portion to TEST |
| **Source** | Contributed, with permission |
| **Access** | `NOT_COLLECTED` |
| **Permitted use** | Whatever each contributor agrees to — **get it in writing** |
| **Redistribution** | **Assume prohibited** unless a contributor says otherwise |
| **Provenance** | Consent must be explicit, per person |

The one input that is not a public dataset and the highest-value data this
project lacks. LFW gave 0 false positives across 501 people, but LFW is
250×250 press photography and contains nothing large and nothing pristine —
which is exactly what both observed false positives were.

`scripts/prepare/datasets.py` requires a `groups.csv` alongside the photos
(`path,group`, grouped by person or device). Without it every photograph is
rejected as `UNSAFE_GROUP` rather than assumed independent.

**Do not collect these by scraping.** Consent is the point.

---

## 6. LFW — already downloaded

| | |
|---|---|
| **Role** | False-positive probe. Evaluation only |
| **Source** | <https://vis-www.cs.umass.edu/lfw/> — obtained via the `bitmind/lfw` mirror on Hugging Face |
| **Access** | `ACCESSIBLE` — public, no form |
| **Status** | **`DOWNLOADED`** — 500 photographs of 500 distinct people |
| **Permitted use** | Unrestricted for research |
| **Redistribution** | Generally permitted; **not committed here anyway** |
| **Provenance** | Public news and web photography of public figures |

Reproduce with `python scripts/fetch_real_faces.py --count 500`.

**Known defect, found by this pipeline:** one photograph appears under two
different people's names (`Carlos_Beltran` / `Raul_Ibanez`). LFW has
documented labelling errors. The duplicate guard caught it on the first run
and rejected both copies as `UNSAFE_GROUP` — neither identity assignment can
be trusted, so neither may be used.

---

## 7. Generated faces and local clips — already present

| Dataset | Source | Status | Notes |
|---|---|---|---|
| StyleGAN2 stills | `almightyj/person-face-dataset-thispersondoesnotexist` (Kaggle) | `DOWNLOADED` — 5 | Per-dataset Kaggle licence; **check before publishing anything derived** |
| Local test clips | Committed in this repository | `DOWNLOADED` — 2 | Fixtures, used to exercise the video path |

---

## The question that must be answered before training

> **May a model trained on FaceForensics++ and Celeb-DF be published?**

Both EULAs restrict the *data*. Whether they restrict *models derived from
the data* is `UNCLEAR` and differs by dataset. Finding out afterwards is the
expensive order.

Three outcomes and what each means:

| If | Then |
|---|---|
| Weights may be published | proceed as planned |
| Weights may not be published | V4 stays private; the repository ships V3 and the evaluation code |
| Only some datasets restrict weights | train on the permissive subset, evaluate on all of them — evaluation is not derivation |

The third option is worth noting: **evaluating on a sealed dataset creates no
derived work**, so Celeb-DF and DeeperForensics can serve as test sets even
under the strictest reading.

---

## Repository hygiene

Enforced, not intended — `tests/test_dataset_pipeline.py` asserts the first
two:

- `datasets/` is gitignored in full — raw media, crops and manifests
- no file under `datasets/` is tracked by git
- no credentials, no signed URLs and no download tokens are committed
- the **seal** lives at `docs/v4_sealed_groups.txt`, deliberately outside the
  gitignored tree, so cleaning a working directory cannot destroy the record
  of what was held out

---

## Next action

**Request access to FaceForensics++, Celeb-DF v2 and DeeperForensics-1.0, and
accept the DFDC licence — today.** Everything else in this phase is built and
tested; approval is the only item measured in weeks.
