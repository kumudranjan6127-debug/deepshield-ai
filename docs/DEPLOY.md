# Deploying DeepShield

The app is one Flask process serving a static frontend and a 17 MB ONNX
model on CPU. No database, no queue, no GPU. It fits a 512 MB free tier with
headroom, and this document is the short list of things that are different
about a server.

---

## The four things that change

| | Local | Server | Why |
|---|---|---|---|
| OpenCV | either build works | **`opencv-python-headless`** | The desktop build links `libGL`, which a server container does not have. This is the single most common way a deploy of this shape fails, and the error (`ImportError: libGL.so.1`) arrives at import time |
| Bind address | `127.0.0.1` | **`0.0.0.0`** | The local default is deliberate — a development run should not be exposed by accident. A container keeping it is simply unreachable |
| Server | Flask's built-in | **gunicorn** | Flask's is single-threaded and explicitly not for production |
| TLS | none | terminated by the platform | The app never sees a certificate; it reads `X-Forwarded-Proto`, and only when told there is a proxy |

---

## Render

`render.yaml` is committed, so the blueprint flow needs no configuration.

1. Push to GitHub (already done).
2. Render → **New → Blueprint** → pick the repository.
3. It reads `render.yaml`. Deploy.

Or by hand:

```
Build   pip install -r requirements.txt gunicorn
Start   gunicorn --pythonpath backend app:app --workers 1 --threads 4 \
                 --timeout 120 --bind 0.0.0.0:$PORT
Health  /api/health
```

### Environment

| Variable | Value | Why |
|---|---|---|
| `DS_HOST` | `0.0.0.0` | reachable |
| `DS_TRUST_PROXY` | `1` | so `X-Forwarded-Proto` may be believed |
| `DS_FORCE_HTTPS` | `1` | redirects reads, refuses plain writes |
| `DS_RATE_LIMIT` | `20` | 5/min is two scans a minute once upload and analyze are both counted |
| `DS_LOG_JSON` | `1` | one object per line for the log viewer |

`PORT` is set by Render and already read by `config.py`.

**Never set `DS_TRUST_PROXY` without a proxy in front.** Any client could then
forge `X-Forwarded-For` and walk around the rate limit.

---

## Why one worker

Two reasons, and they agree.

**Memory.** The benchmark measured a 260 MB peak on a 1024px image. The free
tier gives 512 MB. A second worker loads its own copy of the model and does
not fit.

**Correctness.** The rate limiter and the concurrency gate are in-process. Two
workers means two independent limits — a client gets double the budget, and
`DS_WORKERS=2` becomes four concurrent analyses on a box sized for two.
`SECURITY.md` records this as a limitation; one worker is what makes it moot.

Threads instead: the concurrency gate already caps analyses at two, so the
spare threads serve health checks and static files while an analysis runs
rather than queueing behind it.

---

## What to expect

| | |
|---|---|
| Cold start | The free tier sleeps after 15 minutes. First request wakes the container and loads the model (~0.4 s on top of boot) |
| Image analysis | ~0.45 s, most of it the occlusion heatmap |
| Video | ~50 ms per sampled frame — 3 s for a 60-second clip |
| Peak memory | ~260 MB |

The health check warms the model: `/api/health` asks the engine what it is,
which loads it. So the first real user request is not the one that pays.

---

## Verifying a deploy

```bash
curl https://<your-app>.onrender.com/api/version
```

```json
{"status": "healthy", "engine": "live", "model": "DeepShield V3-Max",
 "runtime": "ONNX", "device": "CPU"}
```

`"engine": "live"` is the one to check. If it says `simulated`, the model
files did not reach the container and every verdict is a demo placeholder.

Then confirm the hardening survived the proxy:

```bash
curl -sI https://<your-app>.onrender.com/ | grep -i "content-security\|strict-transport"
```

`Strict-Transport-Security` only appears when the app can tell the request
arrived over TLS — which means `DS_TRUST_PROXY` is working. If it is missing,
that variable is not set.

---

## If it fails

| Symptom | Cause |
|---|---|
| `ImportError: libGL.so.1` | `opencv-python` instead of `opencv-python-headless` |
| Deploy succeeds, nothing responds | `DS_HOST` still `127.0.0.1` |
| Every request 400 `INSECURE_REQUEST` | `DS_FORCE_HTTPS=1` without `DS_TRUST_PROXY=1` |
| `"engine": "simulated"` | `models/deepshield.onnx` missing — check it is committed and not caught by `.gitignore` |
| Killed during analysis | More than one worker, or a very large upload |
| 429 after a few uploads | `DS_RATE_LIMIT` still at its local default of 5 |

---

## Repository weight

67 MB is tracked, and 46 MB of that is not needed to serve:

| File | Size | Needed at runtime? |
|---|---|---|
| `models/deepshield.onnx` | 17 MB | **yes** |
| `models/face_detection_yunet.onnx` | 228 KB | **yes** |
| `models/deepshield_mobilenetv3.pth` | 17 MB | no — the torch fallback, and torch is not installed |
| `models/archive/*.pth` | 29 MB | no — kept for history |

Not a blocker: the clone is a one-off cost and disk is not the constraint on
these tiers. Worth knowing before assuming the deploy is 17 MB.

---

## Railway

Works with no code change — the committed `Procfile` is what Railway reads.
Railway does not read `render.yaml`, so the environment has to be set by hand:

```
DS_HOST=0.0.0.0
DS_TRUST_PROXY=1
DS_FORCE_HTTPS=1
DS_RATE_LIMIT=20
DS_LOG_JSON=1
```

`PORT` is injected by Railway and already read by `config.py`.

### The cost question, with the numbers

Railway bills usage rather than granting hours. At the published rates
(**$10 / GB / month** for RAM) and this app's measured **~140 MB idle
footprint**:

```
0.14 GB x $10/month  =  ~$1.40/month, idle
Free plan credit     =   $1.00/month
```

Railway's own documentation on credit exhaustion:

> *"if your credit balance reaches zero, your subscription will be cancelled
> ... you will no longer be able to deploy to Railway"*

So the **Free plan runs out in roughly three weeks** and stops, sooner with
traffic. It is not a place to leave something running.

| | Railway Free | Railway Hobby ($5/mo) | Render Free |
|---|---|---|---|
| Runs for | ~20 days, then stops | indefinitely | indefinitely |
| Sleeps when idle | no | no | after 15 min |
| First visit | instant | **instant** | 30–60 s cold start |
| Card required | no | yes | no |
| RAM cap | 0.5 GB | 48 GB | 0.5 GB |

The app peaks at 260 MB, so 0.5 GB is enough everywhere.

**Which to pick.** If $5/month is acceptable, Railway Hobby is the better
demo: the Render free tier's cold start is 30–60 seconds, and the worst
moment for that is someone opening the link in front of an audience. If it
has to be free, Render is the one that keeps running.

Railway's Free plan is the option to avoid — three weeks of uptime followed
by a manual restart every month is worse than either alternative.

## Other platforms

The `Procfile` covers anything that reads one (Heroku-likes, Fly, Koyeb).
The same four rules apply everywhere: headless OpenCV, bind `0.0.0.0`,
gunicorn, one worker.

Fly.io and Docker platforms need a Dockerfile, which this repository does not
have — deliberately. The original constraint was "runs on a laptop with two
commands", and adding container tooling for a single-process Flask app buys
nothing here.
