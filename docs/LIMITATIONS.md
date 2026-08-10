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
- **The false-positive rate has never been measured at scale.** The only
  authentic subject ever scored through the deployed pipeline is one person
  (§2). An accusation against a real photograph is the expensive error, and
  its frequency is unknown.

The general reason: a confident number from an uncalibrated model is a
ranking wearing a percentage's clothes. The interface says "detection
confidence", not "probability", for exactly this reason.

---

## 2. The false-positive rate is unmeasured

Every headline figure is accuracy on data related to training: 99.90%
validation, 99.18% robust, 100% on a held-out StyleGAN2 set. None of them
says how often an authentic photograph is called fake.

Measuring it needs genuine photographs scored through the deployed pipeline.
The repository has one authentic subject — 24 frames of one recording, which
is **one independent observation**, on which no false positive occurred.

Two false positives have been found by hand and fixed:

| Input | Scored | Cause | Fix |
|---|---|---|---|
| 2687px authentic portrait | 0.94 fake | downsampling path unlike training | inputs capped at 1024px |
| Pristine camera original | 0.95 fake | compression domain unlike training | JPEG q88 round trip |

Both were found by inspection, one image at a time. That is not a method.

**What would close this:** a few hundred genuine photographs in
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

## 6. Calibration has never been measured

No ECE, Brier score or reliability curve exists for this model on real data.
`/api/health` reports `"calibrated": false` for that reason.

A network trained with cross-entropy and selected on validation accuracy is
usually **over-confident**: it will report 0.97 on evidence worth 0.80. The
honest assumption is that the percentage overstates the case.

The certainty bands (`very_strong`, `strong`, `uncertain`, `low_evidence`)
are a **vocabulary, not a finding** — the cut points were specified, not
derived from data. And the lowest band is unreachable by construction:
`confidence` is `max(p, 1−p)` for two classes, so it never falls below 50.
Proven by scoring 4,000 random predictions, not argued.

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
