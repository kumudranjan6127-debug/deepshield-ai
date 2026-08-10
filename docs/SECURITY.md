# Security

DeepShield accepts arbitrary files from strangers, decodes them with C
libraries, and fetches URLs on the caller's behalf. Those three sentences
describe the entire threat model.

Everything below is enforced in code and asserted by `tests/test_security.py`
(76 tests). Where a defence exists but was never tested, that is called out —
it happened once already, and it is the dangerous shape: code that is correct
today, protected by nothing that would notice if it stopped being correct.

---

## 1. Uploads — an extension is not evidence

A file called `photo.png` is a PNG only if its bytes say so and a decoder
agrees. Each rung of this ladder exists because skipping it hands
attacker-controlled bytes to a decoder.

```
Content-Length      refused by Werkzeug before the body is buffered
   ↓
extension           .jpg .jpeg .png .webp .mp4 .mov .webm — nothing else
   ↓
declared MIME       checked, never trusted on its own
   ↓
magic bytes         the first 32 bytes must match the claimed format
   ↓
decoder             Pillow / OpenCV must actually open it
   ↓
dimensions          ≤ 40 MP, and large enough to hold a face
   ↓
duration            ≤ 300 s for video
```

Refused, with the code returned:

| Attack | Code |
|---|---|
| `.exe` renamed `.png` | `BAD_MAGIC` |
| HTML renamed `.jpg` | `BAD_MAGIC` |
| ZIP renamed `.mp4` | `BAD_MAGIC` |
| PDF renamed `.png` | `BAD_MAGIC` |
| Valid PNG header, truncated body | `CORRUPT_MEDIA` |
| Valid mp4 header, garbage payload | `CORRUPT_MEDIA` |
| Empty file | `EMPTY_FILE` |
| Over the size cap | `TOO_LARGE` / 413 |
| **Decompression bomb** — kilobytes on disk, hundreds of megapixels in RAM | `IMAGE_TOO_LARGE` |

The bomb case is checked twice over: our own pixel cap fires first, and
Pillow's own guard catches anything larger. Both paths end in a clean 400,
never a 500.

---

## 2. SSRF — the URL feature is the dangerous one

Given a URL, the server fetches it *from inside whatever network it runs in*.
The classic target is cloud metadata at `169.254.169.254`, which hands out
credentials to anyone who asks.

```
scheme        https only (http needs DS_ALLOW_HTTP, off by default)
   ↓
DNS           the hostname is resolved first
   ↓
every answer  ALL resolved addresses must be public — not just the first
   ↓
redirects     each hop re-validated, max 3
   ↓
download      size-capped, content-type checked
```

Blocked: loopback, `0.0.0.0`, private ranges (10/8, 172.16/12, 192.168/16),
link-local (169.254/16), reserved, multicast, and the IPv6 equivalents —
`::1`, `fe80::/10`, `fc00::/7`, and `::ffff:` mapped addresses.

**A hostname blocklist would not be enough.** `evil.example` is an ordinary
name whose DNS answer can point inside the network; only resolving first and
judging the *address* catches it. A host answering with one public and one
internal address is refused outright, because taking the first answer is a
race rather than a check.

Non-HTTP schemes (`file:`, `ftp:`, `gopher:`, `data:`, `javascript:`) are
refused before anything else runs.

**Known gap:** the streaming-platform refusal (YouTube, Instagram, TikTok)
lives in `frontend/assets/js/pages/upload.js` and has no backend equivalent.
Not a hole — those are public hosts and SSRF protection still applies — but an
API caller reaches the network and gets a slow generic failure where the
documentation promises a fast explained one. `KNOWN_ISSUES.md` #5.

---

## 3. Path traversal

`uploadId` is attacker-controlled and becomes a filesystem path.
`os.path.basename` keeps a crafted id inside the upload directory, and static
files go through `send_from_directory`, which Werkzeug keeps inside its root.

Both defences were correct for months and **neither was tested**. Thirteen
traversal cases now cover them — `../`, `..\\`, `....//`, URL-encoded,
absolute POSIX and Windows paths — including one aimed at a sentinel file
that genuinely exists, so a pass cannot be explained away by the target
simply being absent.

---

## 4. Traffic

| Control | Default | Env |
|---|---|---|
| Rate limit | 5 requests / 60 s per client | `DS_RATE_LIMIT`, `DS_RATE_WINDOW` |
| Concurrency | 2 analyses at once, then 503 `BUSY` | `DS_WORKERS`, `DS_QUEUE_WAIT` |
| Upload sweep | staged files deleted after 30 min | `DS_UPLOAD_TTL` |

The rate limit is a sliding window, per client key, in-process. `X-Forwarded-For`
is consulted **only** when `DS_TRUST_PROXY` says there is a proxy in front —
otherwise any client could forge it and walk around the limit.

Inference is CPU-bound and holds ~200 MB, so the concurrency gate is what
stops a burst from turning into thrashing. A failure inside the gate still
releases its slot; without that, the server would stop accepting work after
the first error.

An analysed file is deleted as soon as its verdict is returned. The sweep is
for uploads that were staged and then abandoned.

---

## 5. Response hardening

Every response — API, static page and error alike:

```
Content-Security-Policy   default-src 'self'; script-src 'self';
                          object-src 'none'; frame-ancestors 'none';
                          img-src 'self' data: blob:; base-uri 'self';
                          form-action 'self'
X-Content-Type-Options    nosniff
X-Frame-Options           DENY
Referrer-Policy           strict-origin-when-cross-origin
Permissions-Policy        camera=(), microphone=(), geolocation=(),
                          payment=(), usb=()
Cross-Origin-Opener-Policy      same-origin
Cross-Origin-Resource-Policy    same-origin
Strict-Transport-Security only when the request arrived over TLS
```

The CSP allows **no inline script and no remote origin**. This app renders
user-supplied filenames and server error strings; blocking inline script
turns a future escaping mistake into a blocked request instead of a stolen
session. `data:` is permitted for images because the occlusion heatmap is
delivered as a data URL — it cannot execute.

A test asserts the frontend contains nothing the policy would block, because
a policy the app violates is a policy someone eventually switches off. It has
already caught one violation.

HSTS is sent only over TLS: on a plain local run it would pin `localhost` to
https in the developer's browser and be a nuisance to undo.

---

## 6. CORS

**Never `*`.** This API accepts uploads and fetches URLs on the caller's
behalf; a wildcard would let any page on the internet drive it.

`DS_CORS_ORIGINS` is a comma-separated allow-list, empty by default, because
a same-origin deployment needs no CORS at all. An origin not on the list gets
no CORS headers — the browser refuses the response, which is the correct
outcome rather than an error to report. Preflights are answered before any
route or rate limit sees them.

---

## 7. HTTPS

`DS_FORCE_HTTPS` redirects `GET`/`HEAD` with a 308 and **refuses writes** with
`INSECURE_REQUEST` — a redirect would drop the request body. Whether a request
arrived over TLS is read from `X-Forwarded-Proto` only when `DS_TRUST_PROXY`
is set; otherwise any client could claim TLS by sending a header.

TLS termination itself is the proxy's job. This app is designed to sit behind
one.

---

## 8. What is logged, and what is not

Logged: method, path, status, duration, a per-request id, and the error code
for refusals.

**Not logged:** media, file contents, or verdicts tied to a person. Feedback
is a thumbs up/down and the verdict it refers to — no media, nothing personal.

Unexpected failures return an `incident` id matching a logged traceback. The
user is told an id rather than a reason, because the reason may contain a path
or a filename; the id is what ties their report to the log without asking them
to reproduce anything.

`DS_LOG_FILE` adds a rotating handler (5 MB × 3) — unbounded logs on a small
host eventually fill the disk and take the service with them. `DS_LOG_JSON`
emits one object per line for a collector.

---

## 9. Reporting a problem

Open an issue at
<https://github.com/kumudranjan6127-debug/deepshield-ai/issues>. This is a
student project with no bounty and no SLA; please do not use it for anything
where a compromise would matter.

## 10. What this project does not defend against

- **Authentication.** Firebase auth is optional and the demo mode accepts
  any credentials. There is no authorisation model and no per-user isolation.
- **Multi-process deployment.** The rate limiter and concurrency gate are
  in-process. Two workers means two independent limits.
- **A determined attacker with time.** Nothing here has been penetration
  tested by anyone but its author and its own test suite.
- **The model itself.** Adversarial examples crafted against MobileNetV3 have
  not been considered at all.
