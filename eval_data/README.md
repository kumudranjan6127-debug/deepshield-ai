# Evaluation data

Drop labelled images here and `scripts/evaluate.py` reports every Phase 4B
metric. Nothing in this folder is committed except this file — the datasets
are large, most are licensed, and none of them belong in a git history.

```
eval_data/
  real/<source>/*.jpg
  fake/<source>/*.jpg
  groups.csv          optional: path,group  (identity separation)
  predictions.csv      written by evaluate.py
```

The folder under `real/` or `fake/` names the **source**, and the source is
what the per-source table reports on. Name them however you like; the names
below are only what the rest of the documentation refers to.

---

## The matrix

### REAL — where false positives come from

| Folder | What it is | Where to get it |
|---|---|---|
| `real/photos` | ordinary photographs | your own camera roll, [FFHQ](https://github.com/NVlabs/ffhq-dataset), [CelebA-HQ](https://github.com/tkarras/progressive_growing_of_gans) |
| `real/phone` | straight off a phone camera | your own — this is the case the app actually serves |
| `real/screenshot` | screenshotted, so rescaled and re-encoded | generate, below |
| `real/social` | through a messaging app's compression | generate, below |
| `real/reencode` | forwarded and re-saved repeatedly | generate, below |

The last three are produced from `real/photos` by:

```bash
python scripts/evaluate.py --conditions eval_data/real/photos --out eval_data/real
```

which writes `orig`, `phone`, `screenshot`, `social` and `reencode` variants
of every image. They are stand-ins with plausible parameters — a 720px
q60 JPEG is *like* what a messaging app does, not a measurement of any
particular one. Variants of one photograph share a group, so five files
still count as one independent sample.

### FAKE — the manipulation families

| Folder | Family | Source |
|---|---|---|
| `fake/sg1` | StyleGAN | [140k Real vs Fake Faces](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces) |
| `fake/sg2` | StyleGAN2 | [thispersondoesnotexist set](https://www.kaggle.com/datasets/almightyj/person-face-dataset-thispersondoesnotexist) |
| `fake/dfdc` | face-swap | [DFDC face crops](https://www.kaggle.com/datasets/dagnelies/deepfake-faces) |
| `fake/diffusion` | diffusion | [SD face set](https://www.kaggle.com/datasets/mohannadaymansalah/stable-diffusion-dataaaaaaaaa) |
| `fake/unseen` | anything the model never trained on | whatever is newest |

`fake/unseen` is the one that decides whether this project is honest. A
detector's in-domain score says how well it memorised the generators it
was shown; the unseen score is what a user gets.

---

## Identity separation

`groups.csv` says which images must be counted together:

```csv
path,group
eval_data/fake/dfdc/aagfhgtpmv.jpg,zzzsrjbxej
eval_data/fake/dfdc/bbhtdfuqxq.jpg,zzzsrjbxej
eval_data/real/dfdc/zzzsrjbxej.jpg,zzzsrjbxej
```

Three files, one face — the two fakes were both built from that real video.
Without this they look like three independent tests of the model; with it
they are one. DFDC's own `metadata.csv` has the `original` column that
produces this mapping.

Without a `groups.csv`, the group defaults to the filename stem with any
`__cond-*` suffix stripped.

---

## Running it

```bash
# score everything and write predictions.csv
python scripts/evaluate.py

# in-domain vs never-seen
python scripts/evaluate.py --seen sg1,sg2,dfdc,photos

# how much recall a 1% false-positive budget costs
python scripts/evaluate.py --target-fpr 0.01

# re-score a Kaggle run's predictions with the same arithmetic
python scripts/evaluate.py --from-csv predictions_all.csv --seen 140k_test,tpdn,dfdc
```

A set with only one class is not an error — you simply get the metrics that
class supports. Five StyleGAN2 faces give you a detection rate and nothing
else, and the report says `n/a` for false-positive rate rather than
inventing one.
