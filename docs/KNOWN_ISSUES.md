# Known Issues

Every entry below has been **observed**, not suspected. Where a number is
given, it was measured. Last reviewed after Phase 8, 2026-08-10.

Severity: 🔴 wrong output or wrong information · 🟡 works but flawed ·
🔵 limitation to document rather than fix

---

## 🟡 1. The false-positive rate is measured on press photos, not phone photos

**Resolved in part.** 0 false positives across 501 distinct people (LFW),
95% upper bound 0.60%. Deliberately not FFHQ, which is the model's own
training real class. See `BENCHMARK.md` §1.

What remains: LFW is 250x250 press photography carrying 2000s web
compression. The app receives photographs off a modern phone, and **both
false positives ever found by hand were exactly that kind of image** — a
2687px portrait at 0.94 and a pristine camera original at 0.95. LFW would
have caught neither, because its images are neither large nor pristine.

**Fix:** a few hundred ordinary phone photographs in `eval_data/real/photos`,
then `python scripts/evaluate.py --target-fpr 0.01`.

## 🟡 2. Calibration has never been measured

The UI shows a percentage. Nothing has ever checked whether that
percentage behaves like one — whether images scored 90 are fake about 90%
of the time. Expected Calibration Error, Brier score and a reliability
curve are all unknown for V3.

This matters because a network trained with cross-entropy and selected on
validation accuracy is usually **over-confident**: it will report 0.97 on
evidence worth 0.80. That is why Phase 5 replaced probability language
with evidence language, and why `/api/health` reports
`"calibrated": false`.

`scripts/evaluate.py` prints ECE, MCE, Brier and a reliability diagram for
any labelled set. It needs the same data as [issue 1](#1).

**Fix:** measure it. If the model turns out over-confident, temperature
scaling on a held-out split is a few lines and does not require retraining.

## 🟡 3. The lowest certainty band can never be reached

`CERTAINTY_BANDS` spans 0-100, but `confidence` is `max(p, 1-p) * 100` for
two classes, so it is **never below 50**. "Low evidence" (0-30) is
unreachable by construction, and "Uncertain" (30-70) can only ever hold
the top half of its range. Proven, not suspected —
`tests/test_metrics.py` scores 4,000 random predictions and the bottom
band stays empty.

**And now measured on real data:** across 592 scored images, 591 landed in
`very_strong` and one in `strong`. `uncertain` and `low_evidence` were both
empty. Three of the four bands are unused — two impossible by construction,
one unused in practice, because the model answers roughly 0.02 or 0.97 and
almost nothing between.

The band table was specified against `confidence`. Two ways out, both
deferable until there is data:

- keep four bands and move the cut points into 50-100, or
- band on evidence strength, `|2p - 1| * 100`, which does span 0-100.

**Fix:** pick one once `scripts/evaluate.py` has measured accuracy per
band on real data — the occupancy column makes the answer obvious.

## 🟡 4. The video aggregation has never been validated

Frame scores are combined as `0.40 median + 0.25 mean + 0.35 top-k`, with a
frame counted suspicious at 0.70. Those four numbers were reasoned, not
fitted: no labelled video set has been scored, so nothing has ever checked
that this combination beats a plain mean on real clips.

The weighting deliberately leans on the median, because a false accusation
costs more than a missed forgery. That lean has a measurable cost — a clip
with roughly a third of its frames strongly flagged still comes out "real".
The response reports `suspiciousFrames` and the suspicious timestamps
regardless of the verdict, so the evidence is visible even when the score
does not cross, but that is a mitigation and not a fix.

The same gap covers the four temporal signals: face position jitter, face
size jitter, landmark jitter and appearance continuity are computed and
displayed, and none of them is allowed to affect the verdict, because there
is no evidence for what value of any of them means manipulation.

`tests/test_video.py` pins the behaviour the weights produce, which is not
the same as showing the weights are right.

**Fix:** score a labelled video set — DFDC clips are the obvious source —
then fit the weights and the suspicious threshold against it, and check
whether any temporal signal separates real from fake well enough to earn a
vote.

## 🟡 5. The streaming-platform refusal is browser-only

Pasting a YouTube, Instagram or TikTok link gets a clear explanation — that
those pages are HTML, and that their re-encoding destroys the artefacts
detection depends on. That list lives in `frontend/assets/js/pages/upload.js`
and **nothing equivalent exists in the backend**.

Call `/api/analyze` directly with a YouTube URL and the server resolves the
host, connects, downloads the page, and only then fails on the content. Not
a security hole — SSRF protection still applies and youtube.com is a public
host — but it is a slow, generic failure where the documented behaviour is a
fast, explained one, and it spends outbound bandwidth to learn nothing.

Found by Phase 7: a test that posted a real TikTok URL took **21 seconds**,
because it was genuinely reaching the network.

**Fix:** move the host list into `config.py` and check it in
`security.validate_url`, so both callers get the same refusal. The frontend
list then becomes an optimisation rather than the only guard.

## 🟡 6. `ds_feedback` is not registered in `DS.KEYS`

`DS.KEYS` lists `ds_user`, `ds_scan`, `ds_history`, `ds_settings`. The
feedback store is written as a raw string in three places
(`pages/result.js` ×2, `pages/dashboard.js` ×1). It works, but it sits outside
the convention every other key follows, and "Clear history" does not clear it.

**Fix:** add `FEEDBACK: 'ds_feedback'` to `DS.KEYS` and use it.

## 🟡 7. OpenCV's DNN engine crashes on large batches

Discovered while building the occlusion heatmap: a 36-image batch killed the
process outright (no exception, no traceback). Batches of 4, 8, 12 and 16 were
fine. `_forward()` now chunks at 8, which is a working guard rather than a
diagnosis — the underlying limit in OpenCV 5.0.0.93 is unknown.

## ✅ 8. Face detection failure is silent — FIXED

When YuNet found no face, the whole frame was analysed and the verdict came
back as if a face had been found. The user was never told, so a landscape or
a screenshot produced confident, meaningless output that looked exactly like
a real verdict.

`analyze_file` now returns `faceFound`, the result page carries a "No face
detected" note beside the verdict, and the downloadable report says it in
prose — a report is what gets forwarded to someone else, so the caveat has
to travel with it. Scoring the whole frame is kept as the fallback; the
alternative is refusing landscapes outright, and the honest fix was saying
so rather than hiding it.

## ✅ 8b. Only the largest face in an image was scored — FIXED

Found while investigating a face-swap photograph the app called real.

`_detect_face` picked the largest detection and the image path scored that
one alone. Every other face was discarded without a word. **A group photo
with one swapped face was therefore decided by whichever head happened to be
a few pixels wider** — which is the commonest real deepfake there is, and a
coin flip in a two-person picture.

`_detect_faces` now returns every face (largest first, capped by
`DS_MAX_FACES`, default 6) and the verdict comes from the most suspicious of
them. The max is the only defensible reduction: a photograph containing a
manipulated face is a manipulated photograph, and averaging would let a
crowd outvote the swap, which is exactly the attack. `facesFound` is
reported so the result page can say "4 faces detected" instead of implying
the whole picture was cleared.

Cost is one forward pass per extra face — **29 ms measured locally**, so
about 0.3 s each on the Render free instance.

`tests/test_multiface.py` pins the selection rule with a stubbed classifier.
**A composite image is not a valid probe here**: pasting two faces onto a
canvas changes their resolution and their background, and both move the
score further than the manipulation does. I built one anyway and it produced
a confident wrong story in both directions before I checked the per-face
numbers — see `docs/LIMITATIONS.md`.

## 🟡 9. Verifier reliability is calibrated on very little data

When the verifiers are enabled (`DS_VERIFIERS=1`), their weights and the
"all ≥ 0.85 to overrule" rule were tuned against roughly ten images. Measured
misbehaviour that motivated the current settings:

- SigLIP scored a **re-saved authentic portrait at 1.00 fake**
- ViT scored the same authentic portrait at 0.67 fake
- SigLIP scores 0.00 on StyleGAN faces — it detects diffusion images, so its
  silence is not evidence of authenticity

The default (verifiers off) avoids this entirely, and our own model scores
9/9 alone. Anyone turning them on should re-validate first.

## 🟡 10. Test accuracy is missing from the shipped model metadata

`deepshield.onnx.json` has `"test_accuracy": null`. The 140k test split was
never scored for V3: the Kaggle session was interrupted and the export was
rebuilt from the resume checkpoint, which is written before the test cell
runs. Validation, robust and TPDN figures are present and real.

The same gap is wider than one field: no precision, recall, F1,
specificity, ROC-AUC, PR-AUC or FPR has ever been computed for V3.

**Fix:** score `v3_max.pth` with `scripts/evaluate.py` and patch the JSON
from the result.

## 🟡 11. Training curves for V3 cannot be reproduced

The per-epoch history exists inside the resume checkpoint, but the plotted
`training_curves.png` in `training/results/` is from **V2**. The V3 numbers
are recorded in `MODEL_CARD.md` and were printed live during training.

**Fix:** re-plot from the history array in the resume file, if it is still
available.

## 🟡 12. Kaggle notebooks do not survive a lost session

The Colab notebook checkpoints to Google Drive and resumes cleanly. The Kaggle
version writes to `/kaggle/working`, which is lost when an interactive session
ends without saving a version. The V4 run was cancelled and its progress could
not be recovered.

**Fix:** use *Save Version → Save & Run All (Commit)* for long runs, which
executes headless and persists outputs.

## 🟡 13. Hugging Face downloads stall on some networks

Fetching the verifier models repeatedly froze at ~245 MB. The cause is HF's
Xet transfer protocol; setting `HF_HUB_DISABLE_XET=1` falls back to plain
HTTP and completes at full speed. Documented in the README troubleshooting
table.

## 🟡 14. `DOCUMENTATION.md` predates most of the system

`docs/DOCUMENTATION.md` was written when the project was frontend-only. It
carries a note about the folder restructure, but nothing about the backend,
the model, the ensemble, explainability, the landing page or the ONNX
migration. `CURRENT_STATE.md` and `MODEL_CARD.md` supersede it for anything
factual.

## 🟡 15. Feedback has a path now, but no data in it yet

`POST /api/feedback` now writes to a database when `DATABASE_URL` is set,
and `scripts/analytics.py` reads it — leading with the disagreements, which
are candidate mislabels on real photographs and the closest thing this
project has to the labelled data #1, #2 and #4 are all waiting for.

What is still missing is **traffic**. The table is empty until people use
the deployment and tell it when it is wrong, and feedback is only
collectible going forward — the answers from anyone who used it before the
store existed are gone. See `docs/ANALYTICS.md`.

The dashboard's "user-rated correct" figure is still computed from the
browser's own copy in `localStorage`, so it reflects one browser rather than
everyone.

## 🔵 16. Face-swap deepfakes are not detected

The headline limitation. A real DFDC video scored **97% "real"**. V3 learned
fully generated faces; face-swaps are a different artefact family. Expected
behaviour for this model, not a regression — see `MODEL_CARD.md`.

## 🔵 17. Cross-generator generalisation is not guaranteed

V2 scored thispersondoesnotexist faces at 0.02–0.49 until StyleGAN2 was added
to training; V3 scores the same images 0.97–0.98. A future generator can open
the same gap again. Mitigated by multi-generator training and the optional
verifiers, not solved.

## 🔵 18. Processed media degrades detection

Screenshots, repeated compression and platform re-encoding destroy the
artefacts detection depends on. This is why streaming-platform URLs are
refused with an explanation rather than downloaded. Affects every detector
on the market.

## 🔵 19. Fairness is unmeasured

No evaluation has been run across skin tone, age or gender. The training real
class is FFHQ-only. Nothing here should be read as a fairness claim.

---

## Fixed, kept for the record

| Issue | Resolution |
|---|---|
| V1 learned the dataset's JPEG/resize fingerprint | Anti-shortcut augmentation + robust-based checkpoint selection (V2) |
| StyleGAN2 faces scored 0.02–0.49 | Multi-generator training (V3): the same images now score 0.97–0.98 |
| A 2687px authentic portrait scored 0.94 fake | Inputs capped at 1024px |
| A pristine camera original scored 0.95 fake | JPEG q88 round-trip before inference |
| Averaging buried a confident specialist | Own-led weighting with a corroboration rule |
| Twelve Flask instances accumulated in one session | `scripts/ds.js` works on the port; debug reloader off by default |
| Clearing history left the last report visible | `Clear history` now also clears the session scan |
| Reports page showed only the newest report | "All reports" list added |
| `hidden` attribute lost to `.empty-state { display: flex }` | Global `[hidden] { display: none !important }` |
| Backend needed 1.9 GB to run a 17 MB model | ONNX through OpenCV; 197 MB, verified identical to 3e-7 |
| `/api` reported MobileNetV3-Small / 2.5M / PyTorch for a Large ONNX model | `MODEL_INFO` deleted; identity comes from the model's own metadata (Phase 1), and `tests/test_model_parity.py` now asserts the file, the engine and the API agree |
| Any URL could be fetched, including `127.0.0.1` and cloud metadata | Scheme, DNS and per-redirect address validation (Phase 2) |
| Uploads were accepted on extension alone | Size, MIME, magic bytes, decoder, dimensions and duration (Phase 2) |
| Nothing limited request rate or concurrency | 5/min per client, 2 concurrent analyses, 30-minute upload sweep (Phase 2) |
| The model reported no version | Identity block — `model_name`, `architecture`, `version`, `runtime`, `input_size` (Phase 3) |
| The V4 notebook split DFDC by filename | Several fakes share one source video, and that video is itself in the real class — on synthetic data with the same structure, **78 of 100** held-out files shared a face with training. The split is now by identity group, asserted at runtime and by `tests/test_split.py` (Phase 4) |
| Nothing in the repo could compute a metric | `ds_metrics.py` + `evaluate.py`: accuracy, precision, recall, F1, specificity, ROC-AUC, PR-AUC, FPR, FNR, per-source, threshold sweep — verified by 40 known-answer tests (Phase 4) |
| `tests/test_security.py` aborted on a plain Windows console | A check was named with a `→`, which cp1252 cannot encode; the suite died mid-run. Printed strings are ASCII now (Phase 4) |
| The UI called the heatmap "Grad-CAM attention" | It is occlusion sensitivity, and it measures how much the prediction moves when a region is hidden — not where the network attends. Renamed everywhere it was shown (Phase 5) |
| A confidence percentage read as a probability | The verdict now carries a `certainty` band, and the wording says "detection confidence", with the bands published by `/api/health` so no threshold lives in the browser (Phase 5) |
| Video scores were plain-averaged | Median, mean and a top-k mean are combined, so partial manipulation is no longer diluted and no single frame can carry a verdict; the components ship in the response (Phase 6) |
| The whole `video` block never reached the client | `app.py` forwards only the keys it names and silently dropped it. Fixed, and the regression suite now posts a real clip through the API so a missing block fails a test rather than a demo (Phase 6) |
| Video regression coverage vanished on a fresh clone | `videos()` looked for committed clips and returned an empty list without complaint. It now builds a deterministic clip from the sample faces when the folder is empty (Phase 6) |
| A stale `uploadId` returned a fabricated verdict | Staged uploads are swept after 30 minutes; an expired id fell through to the demo engine and came back 200 OK, carrying the real model's name and no hint that nothing had been analysed. Now `UPLOAD_NOT_FOUND` (Phase 7) |
| Path traversal was defended but never tested | `os.path.basename` on `uploadId` and `send_from_directory` for static files were both correct and both one edit away from a file-read primitive. 13 traversal cases now cover them (Phase 7) |
| The test suites needed a running server | Two of six did, plus `DS_RATE_LIMIT=50` in both places; forgetting either produced failures that looked like bugs. The suite runs on Flask's test client — `python -m pytest`, ~20 s, no server and no network (Phase 7) |
| Test media had to be remembered | Every fixture is now generated from repository material by `tests/conftest.py`, so coverage cannot quietly vanish on a fresh clone (Phase 7) |
| Every page displayed MobileNetV3-Small / 2.5M / PyTorch as its fallback | Phase 1 deleted the stale constant from the backend but left the same values hardcoded in eight pages, so with no backend answering the UI described a model that has never been in production. All of it now reads from `/api/health` (Phase 8) |
| The dashboard badge said "Simulated (demo)" while a real model was running | The badge was static HTML that only ever upgraded to live; there was no badge at all on the result page, where the verdict is actually read. One painter now serves every page, and each verdict carries its own `engine` field (Phase 8) |
| Analysis failures showed "Analysis failed. Please try again." | The backend produces a specific reason and a stable code for every refusal; the UI discarded both, showed a toast that vanished, and redirected to the dashboard. The reason, the code and a recovery hint now stay on screen (Phase 8) |
| The explanation named one region | The occlusion grid was already scored, so ranking every region it leaned on cost nothing — a face-swap that gives itself away at both the eyes and the mouth now says so (Phase 8) |
| Two tuning knobs cannot be changed at runtime | `predict_video(max_frames=CFG.MAX_VIDEO_FRAMES)` and `explain(grid=CFG.OCCLUSION_GRID)` read config as **default arguments**, which Python evaluates once at import. The environment variables work correctly, but mutating `CFG` in a running process silently does nothing - which produced two wrong benchmark conclusions before it was noticed. Documented in `DEPLOY.md`; measuring these requires a fresh process |
