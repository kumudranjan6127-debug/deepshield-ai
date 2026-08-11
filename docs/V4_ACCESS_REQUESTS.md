# Dataset access requests — the 10 minutes that unblock three weeks

These forms take minutes to submit and days-to-weeks to be approved. Nothing
else on the V4 path takes that long, so they go first even though they pay
off last.

**Fill these in as you go.** An empty `Requested` column three weeks from now
is the most likely reason V4 has not started.

| Dataset | Where | Needs | Requested | Approved |
|---|---|---|---|---|
| FaceForensics++ | <https://github.com/ondyari/FaceForensics> — the request form linked in the README | Institutional email helps; a one-line project description | ☐ | ☐ |
| Celeb-DF v2 | <https://github.com/yuezunli/celeb-deepfakeforensics> — email form in the README | Same | ☐ | ☐ |
| DeeperForensics-1.0 | <https://github.com/EndlessSora/DeeperForensics-1.0> | Agreement form | ☐ | ☐ |
| DFDC | <https://ai.meta.com/datasets/dfdc/> or the Kaggle competition page | Licence acceptance only — **no waiting** | ☐ | n/a |

## What to write in the form

Keep it short and true. Something like:

> Undergraduate final-year project building a CPU-only deepfake detector
> (MobileNetV3, ONNX). I need FaceForensics++ to train on face-swap and
> reenactment manipulations, which my current model does not detect. Research
> and educational use only; the dataset will not be redistributed.

Do not claim an affiliation you do not have. A student project is a legitimate
academic use and saying so is fine.

## The question to settle while you wait

> **May a model trained on this dataset be published?**

Ask it in the same email. Both EULAs clearly restrict the *data*; whether they
restrict *models derived from it* is unclear, and finding out after training
is the expensive order. See `V4_DATASET_PROVENANCE.md`.

If the answer is no for FF++ or Celeb-DF, V4 can still be trained and
evaluated — it just cannot be published, and the repository ships V3 plus the
evaluation code instead.

## You do not have to wait to start

`dagnelies/deepfake-faces` on Kaggle is **DFDC face crops, public, no
approval**. Together with the four generated-face sets already wired into the
V4 notebook, that is enough to train a face-swap-capable V4 this week.

What the approved datasets add later:

| Dataset | What it buys |
|---|---|
| FaceForensics++ | Four named manipulation methods and real H.264 compression levels — and the withheld method for the cross-manipulation test |
| Celeb-DF v2 | The honest cross-dataset number. Nothing else can provide it |
| DeeperForensics | A robustness curve rather than a single number |

So: submit the forms today, then start V4 on the public data without waiting
for any of them.
