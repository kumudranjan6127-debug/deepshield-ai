# Limitations

What this system cannot do, stated plainly. Nothing here is speculative:
every entry has been observed, and where a number appears it was measured.

If you read one section, read §1.

---

## 1. Do not use this to decide anything about a person

DeepShield is a screening aid built as a student project. It has never been
validated for forensic use, legal evidence, journalism, moderation, or any
decision that affects someone.

Two specific reasons, before the general one:

- **A real DFDC deepfake video scored 97% "real".** Face-swaps are the common
  video deepfake and the model does not detect them (§3).
- **The false-positive rate is measured on one condition only.** Zero false
  positives across 501 press photographs (§2) — but the app receives photos
  off a phone, and that condition has never been scored. Both false positives
  ever found by hand were exactly that kind of image.

The general reason: a confident number from an uncalibrated model is a
ranking wearing a percentage's clothes. The interface says "detection
confidence", not "probability", for exactly this reason.

---

## 2. The false-positive rate is measured, on one condition only

**0 false positives across 501 distinct people.** 95% upper bound **0.60%**
(rule of three). The real set is LFW — press and web photographs, one per
person, deliberately not FFHQ because FFHQ is what the model trained on.

The two distributions do not overlap: the highest-scoring real photograph is
0.1074, the lowest-scoring fake is 0.9689.

**What is still unknown.** LFW is 250×250 press photography carrying 2000s
web compression. The app receives photographs off a modern phone — higher
resolution, different sensor noise, different compression history — and that
condition has never been scored. The two false positives ever found by hand
were both of exactly that kind:

| Input | Scored | Cause | Fix |
|---|---|---|---|
| 2687px authentic portrait | 0.94 fake | downsampling path unlike training | inputs capped at 1024px |
| Pristine camera original | 0.95 fake | compression domain unlike training | JPEG q88 round trip |

Both are fixed, and LFW would not have caught either — its images are neither
large nor pristine. A phone-photo set remains the most valuable data this
project does not have.

**What would close it:** a few hundred ordinary phone photographs in
`eval_data/real/photos`, then `python scripts/evaluate.py --target-fpr 0.01`.

---

## 3. Face-swap deepfakes are not detected

V3 learned **fully generated** faces — StyleGAN, StyleGAN2, diffusion. A
face-swap leaves a different artefact family: blending seams and boundary
inconsistencies rather than generator fingerprints.

A real DFDC video was scored **97% real**.

This is expected behaviour for this model, not a regression. `V4-Universal`
exists to close it — the notebook trains on DFDC face crops with
identity-safe splitting — and has never been run to completion.

---

## 4. A new generator can open the gap again

V2 scored thispersondoesnotexist faces at **0.02–0.49** until StyleGAN2 was
added to training; V3 scores the same images 0.97–0.98. A generator released
after this model was trained can reopen exactly that gap, and nothing in the
system would announce it.

Mitigated by multi-generator training, not solved. **Cross-dataset accuracy —
performance on a generator never trained on — has never been measured**, so
the size of the gap today is unknown.

---

## 5. Processing destroys the evidence

Screenshots, repeated compression and platform re-encoding remove the
high-frequency artefacts detection depends on. This affects every detector on
the market, not just this one.

Measured on five StyleGAN2 faces, detection survived phone, screenshot,
messaging-app and re-encode conditions with the score moving at most 0.006 —
but that is five images of one generator, and it says nothing about the real
class.

It is why streaming-platform links are refused with an explanation instead of
being downloaded and scored.

---

## 6. Calibration is measured, and the shape of it is the problem

Measured on 592 images: **ECE 0.0242, MCE 0.1074, Brier 0.0006**. Those look
excellent, and on this data they are — but they are the calibration of a model
whose outputs are almost all 0.02 or 0.97. `/api/health` still reports
`"calibrated": false`, because a reliability curve with two occupied bins has
not demonstrated calibration anywhere in between.

**591 of 592 verdicts land in the `very_strong` band.** A four-band vocabulary
describes a distribution this model does not have, and a detector this
confident on every input will be just as confident when it is wrong.

A network trained with cross-entropy and selected on validation accuracy is
usually over-confident. This one is not, on this data — but this data never
put it in a difficult position. Every image was easy.

The certainty bands (`very_strong`, `strong`, `uncertain`, `low_evidence`)
are a **vocabulary, not a finding** — the cut points were specified, not
derived from data. Two of the four are unreachable by construction —
`confidence` is `max(p, 1−p)` for two classes, so it never falls below 50 —
and the measurement above shows a third goes unused in practice. One band
carries 591 of 592 verdicts.

---

## 7. Fairness is unmeasured

No evaluation has been run across skin tone, age or gender. The training real
class is **FFHQ only** — a single, curated, aligned source.

Nothing in this project should be read as a fairness claim. A detector whose
real class comes from one dataset has an obvious route to behaving worse for
people that dataset under-represents, and nobody has checked.

---

## 8. Video is still frame-wise

The classifier sees single frames. It cannot see flicker, blink rate or
lip-sync desync — which is where most video deepfakes actually give
themselves away.

Frames are combined with median, mean and a top-k mean rather than a plain
average, so a partly manipulated clip is no longer diluted. But **those
weights have never been fitted** against labelled video, and the deliberate
lean toward the median has a cost: a clip with roughly a third of its frames
strongly flagged still comes out "real". The response reports
`suspiciousFrames` and the suspicious timestamps regardless of the verdict,
so the evidence is visible — that is a mitigation, not a fix.

The four temporal signals are reported and **never counted**, because no
evidence exists for what value of any of them means manipulation.

---

## 9. No face means no warning

When YuNet finds no face, the whole frame is analysed and the verdict is
returned as though a face had been found. For a landscape, a document or a
crowd this produces a confident, meaningless answer.

The detector *knows* — `_detect_face` returns `found: False` — and the API
does not pass it on. `KNOWN_ISSUES.md` #7. A test pins the current behaviour
so that adding a `faceFound` flag fails loudly rather than changing the
contract quietly.

---

## 10. Operational limits

| Limit | Value | Consequence |
|---|---|---|
| Concurrency | 2 analyses | The third caller waits, then gets a 503 |
| Rate limit | 5 requests / minute / client | In-process: two workers means two independent limits |
| Video length | 300 s | Longer uploads are refused |
| Frames sampled | 60 maximum | A 10-minute clip is judged on 60 frames |
| Upload size | 100 MB | |
| Image size | 40 megapixels | |
| Peak memory | ~260 MB | One process; two would not fit a 512 MB tier |

There is no authentication model. Firebase auth is optional and the demo mode
accepts any credentials; there is no per-user isolation and no authorisation
anywhere.

---

## 11. Nothing here has been independently reviewed

No penetration test, no external audit, no reproduction by anyone other than
the author. The test suite (261 tests) and the regression baseline are the
only checks, and both were written by the same person who wrote the code.

Every measured claim in `BENCHMARK.md` can be regenerated from the CSV that
produced it. That is reproducibility, which is not the same as validation.

---

## Where these are tracked

`KNOWN_ISSUES.md` carries all of them with severities and current status,
including the ones already fixed. This document is the readable summary; that
one is the ledger.
