# Analytics and feedback

## Why this exists

The three largest open issues in this project all want the same thing:

| | |
|---|---|
| `KNOWN_ISSUES #1` | the false-positive rate is measured against press photography, never phone photographs |
| `#2` | calibration is measured on images the model found easy |
| `#4` | the video combiner weights were reasoned, never fitted |

All three are waiting on **real images with someone saying whether the
answer was right**.

`POST /api/feedback` has collected exactly that since Phase 2. It wrote to a
JSONL file that nothing read (`#15`) — and on a free host that file is erased
every time the service sleeps, which is every fifteen idle minutes. This is
what makes those answers survive.

**A disagreement is a candidate mislabel on a real photograph.** That is the
whole point.

---

## What is stored

| Column | Example |
|---|---|
| `at` | when |
| `scan_id` | the browser's own id for that scan — links a verdict to feedback on it |
| `file_type` / `file_ext` / `file_bytes` | `image`, `.jpg`, `2048` |
| `prediction` / `confidence` / `certainty` / `risk` | `deepfake`, `97`, `very_strong`, `high` |
| `engine` / `model_version` / `runtime` | `live`, `V3-Max`, `ONNX` |
| `frames` / `suspicious` | video only |
| `latency_ms` | how long it took |

Feedback rows carry the same verdict fields plus `agree` and an optional
`note`.

## What is not stored, and why

| Never | Because |
|---|---|
| **Media** — the image, a thumbnail, a hash of one | The app can honestly say your media does not stay on the server. In a deepfake tool that is a feature, not a limitation |
| **Filenames** | `passport.jpg` and `me_and_priya.mp4` are personal in a way a file size is not. Only the extension survives |
| **IP addresses** | The rate limiter holds one in memory for sixty seconds. Writing it down turns analytics into tracking |
| **Anything tying a row to a person** | `scan_id` identifies a scan, not a human |

`tests/test_store.py` asserts each of these absences. Absence is what nobody
notices going missing, so it is what gets tested.

---

## Turning it on

Nothing is required locally. With no `DATABASE_URL` the app behaves exactly
as it did before: feedback appends to JSONL, analyses are not recorded, and
no driver is installed.

For the deployment:

```bash
pip install "psycopg[binary]"
```

then set `DATABASE_URL` to a Postgres connection string. The schema creates
itself on first write.

### Choosing a database

Two verified constraints rule out the obvious options:

> Render's **free Postgres expires 30 days after creation** — 14 days grace,
> then deletion.
> A free web service's **filesystem is wiped on every redeploy, restart and
> spin-down** — which happens after fifteen idle minutes.

So neither Render's own database nor the local disk will hold anything.
Supabase's free tier does not expire (500 MB Postgres, paused after a week
of inactivity, data retained), and Neon is a similar shape. Check the terms
before relying on either — they change.

### Supabase: take the Session Pooler string, not the Direct one

Supabase offers three connection strings and shows the wrong one first.

| Method | Port | IPv4 | Use it? |
|---|---|---|---|
| **Direct connection** | 5432 | **paid add-on only** — IPv6 by default | **No.** This is the one shown first, and on an IPv4-only host it simply will not connect |
| **Session pooler** | 5432 (pooler host) | yes, all tiers | **Yes** |
| Transaction pooler | 6543 | yes | No — *"transaction mode does not support prepared statements"*, and psycopg uses them |

The session-pooler host looks like
`aws-0-<region>.pooler.supabase.com`, and the username carries the project
ref: `postgres.<projectref>`.

A paused project (a week without traffic) refuses connections. That is
survivable by design — the store logs the failure and the app carries on
serving verdicts — but nothing is recorded until it is resumed.

---

## Reading it

```bash
DATABASE_URL=... python scripts/analytics.py
DATABASE_URL=... python scripts/analytics.py --days 7
DATABASE_URL=... python scripts/analytics.py --export disagreements.csv
```

**Deliberately a script, not an endpoint.** An `/api/analytics` route would
be readable by anyone who guessed the path, and gating it would mean building
backend authentication this project does not have. A script run by whoever
holds the connection string needs neither, and adds no attack surface.

Output leads with the disagreements, because those are the rows worth a
person's time.

---

## Two rules for using what it collects

**A disagreement is a claim, not a label.** Someone may simply be wrong, or
testing, or clicking. Read the rows before treating any of them as ground
truth. The script says so every time it prints them.

**Nothing here reaches the model automatically.** This is an evaluation
signal. A pipeline that retrained on unreviewed user feedback would be one
bad actor away from a poisoned model, and the whole point of the sealed test
set is that training data is chosen deliberately.

---

## It cannot take the product down

Analytics that can break a request is worse than no analytics. Every write is
best-effort, runs off the request thread, and swallows its own errors after
logging them — and the call site is guarded as well as the store.

That second guard is not theoretical. While this module was being written, a
missing import turned eleven completed analyses into 500s: the model had
already produced an answer and the answer was discarded because a logging
call failed. `tests/test_store.py` now breaks the store on purpose and
asserts the verdict still arrives.
