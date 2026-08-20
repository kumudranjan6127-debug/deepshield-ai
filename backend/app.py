"""
============================================================
DeepShield AI — Flask backend

    Frontend → API → Validation → Inference → Result
                 │                    │
              errors.py          inference.py
                 └──── config.py ─────┘

This module is routing only: it validates a request, hands it to the
engine, and shapes the reply. Paths and limits live in config.py, failure
shapes in errors.py, and every model fact comes from inference.py — there
are no model constants here.

Run:   venv/Scripts/python backend/app.py   (or: npm start)
Open:  http://localhost:5000
============================================================
"""
import json
import logging
import os
import threading
import time
import uuid
from datetime import datetime, timezone

import errors
import inference
import network
import security
import store
from config import CFG
from flask import Flask, g, jsonify, redirect, request, send_from_directory

log = logging.getLogger("deepshield")

# Traffic controls, shared by every request
limiter = security.RateLimiter(CFG.RATE_LIMIT, CFG.RATE_WINDOW_SECONDS)
gate = security.InferenceGate(CFG.MAX_CONCURRENT_ANALYSES, CFG.QUEUE_WAIT_SECONDS)

# OpenCV's shared DNN net and YuNet detector are mutable. Gunicorn serves this
# app with multiple threads, so every access to the shared inference engine is
# serialized to prevent cross-request setInput/detect races.
engine_access_lock = threading.Lock()

# WSGI servers import app.py; they never execute the __main__ block. Start
# upload cleanup lazily in each serving process on its first request.
_housekeeping_lock = threading.Lock()
_housekeeping_started = False


def client_key() -> str:
    """Who is calling. X-Forwarded-For is only consulted behind a proxy we
    were told to trust — otherwise any client could spoof its identity and
    walk around the rate limit."""
    if CFG.TRUST_PROXY:
        forwarded = request.headers.get("X-Forwarded-For", "")
        if forwarded:
            return forwarded.split(",")[0].strip()
    return request.remote_addr or "unknown"


class JsonLines(logging.Formatter):
    """One JSON object per line, for anything that ships logs somewhere.

    Grep works on the plain format; a log collector wants fields. Which one
    you get is DS_LOG_JSON, and neither changes what is logged."""

    def format(self, record):
        payload = {
            "time": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if getattr(record, "request_id", None):
            payload["request_id"] = record.request_id
        if record.exc_info:
            payload["traceback"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def setup_logging():
    """One configuration for the whole process; modules just get a logger."""
    level = getattr(logging, CFG.LOG_LEVEL, logging.INFO)
    formatter = (JsonLines() if CFG.LOG_JSON else logging.Formatter(
        "%(asctime)s %(levelname)-7s %(name)s  %(message)s", datefmt="%H:%M:%S"))

    handlers = [logging.StreamHandler()]
    if CFG.LOG_FILE:
        # Rotating, because an unbounded log on a small host eventually
        # fills the disk and takes the service with it.
        from logging.handlers import RotatingFileHandler
        handlers.append(RotatingFileHandler(
            CFG.LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3,
            encoding="utf-8"))

    root = logging.getLogger()
    root.setLevel(level)
    for handler in list(root.handlers):
        root.removeHandler(handler)
    for handler in handlers:
        handler.setFormatter(formatter)
        root.addHandler(handler)

    logging.getLogger("werkzeug").setLevel(logging.WARNING)  # per-request noise


CFG.ensure_dirs()
app = Flask(__name__, static_folder=None)
# Refused by Werkzeug before the body is read into memory.
app.config["MAX_CONTENT_LENGTH"] = CFG.MAX_UPLOAD_BYTES
errors.register(app)


def request_is_secure() -> bool:
    """Did this request arrive over TLS?

    Behind a proxy the connection to Flask is plain HTTP, so the only
    evidence is the forwarded header — and that header is only trustworthy
    when we were told there is a proxy in front."""
    if request.is_secure:
        return True
    if CFG.TRUST_PROXY:
        return request.headers.get("X-Forwarded-Proto", "").split(",")[0].strip() == "https"
    return False


def _housekeeping_worker():
    """Best-effort startup sweep plus the periodic cleanup loop."""
    try:
        security.cleanup_uploads()
    except Exception:
        log.warning("initial upload cleanup failed", exc_info=True)
    try:
        security.start_cleanup_thread()
    except Exception:
        log.warning("periodic upload cleanup did not start", exc_info=True)


def _ensure_housekeeping():
    """Start abandoned-upload cleanup once without delaying user requests."""
    global _housekeeping_started
    if _housekeeping_started:
        return
    with _housekeeping_lock:
        if _housekeeping_started:
            return
        # Mark first so failures never turn housekeeping into a per-request
        # retry loop. The potentially slow directory sweep runs off-thread.
        _housekeeping_started = True
        try:
            threading.Thread(
                target=_housekeeping_worker,
                name="upload-housekeeping-init",
                daemon=True,
            ).start()
        except Exception:
            log.warning("upload housekeeping did not start", exc_info=True)


@app.before_request
def ensure_housekeeping():
    _ensure_housekeeping()


@app.before_request
def tag_request():
    """A short id for this request, echoed on failures.

    When someone reports "it said something went wrong", this is what turns
    that into a line in the log."""
    g.request_id = uuid.uuid4().hex[:12]
    g.started = time.perf_counter()


@app.after_request
def log_api_call(response):
    """One line per API call, with the timing already measured."""
    if request.path.startswith("/api/") and request.path != "/api/health":
        log.info("%s %s -> %s in %d ms [%s]", request.method,
                 errors.safe_log_text(request.path),
                 response.status_code,
                 int((time.perf_counter() - getattr(g, "started", time.perf_counter())) * 1000),
                 getattr(g, "request_id", "-"))
    return response


@app.before_request
def enforce_https():
    """Send plain HTTP to the TLS address. Off unless DS_FORCE_HTTPS is set,
    because a local run has no certificate and should not redirect."""
    if not CFG.FORCE_HTTPS or request_is_secure():
        return None
    if request.method not in ("GET", "HEAD"):
        # Redirecting a POST loses the body; refuse plainly instead.
        raise errors.insecure_request()
    return redirect(request.url.replace("http://", "https://", 1), code=308)


@app.before_request
def cors_preflight():
    """Answer OPTIONS before any route or rate limit sees it."""
    if request.method != "OPTIONS":
        return None
    allowed = security.cors_headers(request.headers.get("Origin", ""))
    response = app.make_response(("", 204 if allowed else 403))
    response.headers.extend(allowed)
    return response


@app.after_request
def harden(response):
    """Every response, including static files and errors."""
    for name, value in security.security_headers(request_is_secure()).items():
        response.headers.setdefault(name, value)
    for name, value in security.cors_headers(request.headers.get("Origin", "")).items():
        response.headers[name] = value
    return response


# ---------------------------------------------------------- frontend

@app.route("/")
def home():
    return send_from_directory(CFG.FRONTEND_DIR, "landing.html")


@app.route("/<path:path>")
def assets(path):
    """Static catch-all. frontend/ holds only public files, and
    send_from_directory is path-traversal safe — backend code, models and
    uploads all live outside that root."""
    return send_from_directory(CFG.FRONTEND_DIR, path)


# ---------------------------------------------------------- model identity

def _engine_info() -> dict:
    """Read/load the shared engine without racing another request."""
    with engine_access_lock:
        return inference.engine_info()


def model_identity(info: dict | None = None) -> dict:
    """What is actually loaded, asked of the engine rather than hardcoded.

    With no model present the fields say so rather than describing a model
    that is not running — the UI shows "Simulated (demo)" in that state."""
    info = _engine_info() if info is None else info
    if not info:
        return {
            "model_name": "DeepShield", "architecture": "—", "version": "—",
            "runtime": "simulated", "input_size": None,
            "name": "MobileNetV3", "params": "—", "input": "—",
            "backend": "none", "device": "CPU",
        }

    size = info.get("input_size")
    return {
        # The block Phase 3 specifies
        "model_name": info.get("model_name", "DeepShield"),
        "architecture": info.get("architecture"),
        "version": info.get("version"),
        "runtime": info.get("runtime"),
        "input_size": size,
        # Display fields the existing pages already read
        "name": info.get("architecture"),
        "params": info.get("params") or "—",
        "input": f"{size} × {size}" if size else "—",
        "backend": info.get("runtime", "none"),
        "device": "CPU",
    }


# ---------------------------------------------------------- validation

def staged_upload_path(upload_id: str) -> str | None:
    """Resolve an uploadId to a staged file. basename() keeps a crafted id
    from escaping the upload directory."""
    if not upload_id:
        return None
    path = os.path.join(CFG.UPLOAD_DIR, os.path.basename(str(upload_id)))
    return path if os.path.exists(path) else None


def new_temp_path(suffix: str) -> str:
    return os.path.join(CFG.UPLOAD_DIR, uuid.uuid4().hex + suffix)


def discard(path: str | None):
    """Uploads are temporary by design — analysis never leaves one behind."""
    if not path or not os.path.exists(path):
        return
    try:
        os.remove(path)
    except OSError as e:
        log.warning("could not remove %s: %s", path, e)


# ---------------------------------------------------------- echo engine
# Used when no model is present. Deliberately mirrors DS.api._verdictFor in
# the frontend so a demo gives the same answer in either mode.

def fnv_hash(s: str) -> int:
    """FNV-1a, bit-identical to DS.util.hash in assets/js/utils.js."""
    h = 2166136261
    for ch in s:
        h ^= ord(ch)
        h = (h * 16777619) & 0xFFFFFFFF
    return h


def echo_verdict(file_name, file_size, file_type):
    name = (file_name or "").lower()
    seed = fnv_hash(name + str(file_size or 0))

    if any(k in name for k in ("fake", "synth", "gen")):
        prediction = "deepfake"
    elif any(k in name for k in ("real", "orig")):
        prediction = "real"
    else:
        prediction = "deepfake" if seed % 100 < 42 else "real"

    confidence = 72 + (seed % 26)  # 72–97
    frames = 24 + (seed % 37) if file_type == "video" else 1
    return prediction, confidence, frames


# ---------------------------------------------------------- API

@app.get("/api/health")
def health():
    live = inference.engine_available()
    info = _engine_info() if live else {}
    return jsonify({
        "ok": True,
        "status": "healthy",
        "engine": "live" if live else "echo",
        "model": model_identity(info),
        # The frontend labels confidence numbers with these, rather than
        # keeping its own copy of the thresholds.
        "certainty_bands": inference.certainty_bands(),
        "calibrated": False,   # no reliability curve has ever been measured
        "score_description": CFG.SCORE_DESCRIPTION,
        **info,
    })


@app.get("/api/version")
def version():
    """What is running, in one small object.

    `/api/health` grew to carry everything the frontend needs to render
    itself. This is the short answer for a monitor, a deploy check or a
    person asking "which model is live?" — five fields, no arrays, cheap to
    read and cheap to diff between deployments.

    Every value comes from the model's own metadata; nothing here is
    written down twice."""
    live = inference.engine_available()
    info = _engine_info() if live else {}
    identity = model_identity(info)
    name = " ".join(x for x in (identity.get("model_name"), identity.get("version"))
                    if x and x != "—")
    return jsonify({
        "status": "healthy",
        "engine": "live" if live else "simulated",
        "model": name or "unknown",
        "architecture": identity.get("architecture"),
        "runtime": identity.get("runtime"),
        "device": identity.get("device"),
        "input_size": identity.get("input_size"),
        "classes": info.get("classes"),
        "calibrated": False,
        "score_description": CFG.SCORE_DESCRIPTION,
    })


@app.post("/api/feedback")
def feedback():
    """Record whether the user agreed with a verdict.

    Stores no media and nothing personal — the verdict and a thumbs
    up/down. This is an evaluation signal, never a training label:
    nothing here reaches the model automatically."""
    # Feedback writes to disk and may also spawn a database-write thread.
    # It needs the same abuse protection as uploads and analyses.
    limiter.check(client_key())

    d = request.get_json(silent=True) or {}
    if not isinstance(d.get("agree"), bool):
        raise errors.bad_field("agree", "true or false")

    entry = {
        "scanId": str(d.get("scanId", ""))[:40],
        "prediction": str(d.get("prediction", ""))[:20],
        "confidence": d.get("confidence"),
        "fileType": str(d.get("fileType", ""))[:10],
        "agree": d["agree"],
        "note": str(d.get("note", ""))[:280],
    }
    store.record_feedback(entry)
    log.info("feedback recorded: agree=%s", entry["agree"])
    return jsonify({"ok": True})


@app.post("/api/upload")
def upload():
    """Stage a file for analysis. A File object cannot survive the page
    navigation to processing.html, so the upload page parks it here and
    passes the returned uploadId along.

    Validation runs cheapest-first: size (already enforced by
    MAX_CONTENT_LENGTH), extension, declared type, magic bytes — then the
    file is written and actually decoded before we admit to holding it."""
    limiter.check(client_key())

    if "file" not in request.files:
        raise errors.no_file()
    f = request.files["file"]
    ext, kind = security.validate_upload(f)

    upload_id = uuid.uuid4().hex + ext
    path = os.path.join(CFG.UPLOAD_DIR, upload_id)
    f.save(path)
    try:
        details = security.validate_media_file(path, kind)
    except Exception:
        discard(path)          # never keep something we rejected
        raise

    log.info("staged upload %s (%s, %s)", upload_id, f.filename, details)
    return jsonify({"ok": True, "uploadId": upload_id})


def _read_request(live: bool):
    """Pull the analysis request apart, whichever way it arrived.

    Returns (file_name, file_type, file_size, frame_rate, media_path,
    owned) — `owned` marks a file this request created, which is the only
    kind it may delete afterwards."""
    if "file" in request.files:
        f = request.files["file"]
        file_name = f.filename or "upload"
        frame_rate = float(request.form.get("frameRate", CFG.DEFAULT_FRAME_RATE))
        # The declared kind is cross-checked against the real container
        ext, kind = security.validate_upload(f)
        file_type = request.form.get("fileType", kind)
        if file_type != kind:
            log.info("declared %s but the file is a %s — trusting the file",
                     file_type, kind)
            file_type = kind
        if live:
            path = new_temp_path(ext)
            f.save(path)
            try:
                security.validate_media_file(path, kind)
            except Exception:
                discard(path)
                raise
            return file_name, file_type, os.path.getsize(path), frame_rate, path, True
        return file_name, file_type, len(f.read()), frame_rate, None, False

    d = request.get_json(silent=True) or {}
    url = d.get("url") or ""
    file_name = d.get("fileName") or (
        url.rstrip("/").rsplit("/", 1)[-1] if url else "video.mp4")
    file_type = d.get("fileType", "video")
    file_size = d.get("fileSize")
    frame_rate = float(d.get("frameRate", CFG.DEFAULT_FRAME_RATE))

    if live:
        upload_id = d.get("uploadId")
        staged = staged_upload_path(upload_id)
        if staged:
            return (file_name, file_type, file_size or os.path.getsize(staged),
                    frame_rate, staged, True)
        if upload_id:
            # The caller named media. Falling through from here reached the
            # demo engine, which returned a fabricated verdict carrying the
            # real model's name — a stale id looked exactly like an answer.
            raise errors.upload_not_found()
        if url:
            path = new_temp_path(".mp4")
            size = network.safe_download(url, path)
            try:
                security.validate_media_file(path, "video")
            except Exception:
                discard(path)
                raise
            return file_name, "video", size, frame_rate, path, True

    return file_name, file_type, file_size, frame_rate, None, False


def _run_inference(
    media_path: str, file_type: str, frame_rate: float,
    deadline: float | None = None,
):
    """Call the shared mutable model without exceeding the queue deadline."""
    if deadline is None:
        acquired = engine_access_lock.acquire()
    else:
        acquired = engine_access_lock.acquire(
            timeout=max(0.0, deadline - time.monotonic())
        )
    if not acquired:
        raise errors.server_busy()
    try:
        return inference.analyze_file(media_path, file_type, frame_rate)
    finally:
        engine_access_lock.release()


@app.post("/api/analyze")
def analyze():
    started = time.perf_counter()
    limiter.check(client_key())
    live = inference.engine_available()
    media_path = owned = None

    # The browser's own id for this scan. It links a verdict to the feedback
    # someone later leaves on it, and identifies a scan rather than a person.
    scan_id = str((request.get_json(silent=True) or {}).get("scanId")
                  or request.form.get("scanId") or "")[:64]

    try:
        (file_name, file_type, file_size,
         frame_rate, media_path, owned) = _read_request(live)

        extras = {}
        if live and media_path:
            # Gate wait plus engine-lock wait share one queue budget.
            queue_deadline = time.monotonic() + max(0.0, CFG.QUEUE_WAIT_SECONDS)
            with gate:
                result = _run_inference(
                    media_path, file_type, frame_rate, queue_deadline
                )
            prediction = result["prediction"]
            confidence = result["confidence"]
            frames = result["framesAnalyzed"]
            # Optional blocks the engine may produce. `video` is present for
            # video scans only; the others for images only. Keys the engine
            # did not emit are dropped rather than sent as null.
            extras = {k: result[k] for k in
                      ("ensemble", "disputed", "explain", "video",
                       "faceFound", "facesFound", "insufficientEvidence", "reason",
                       "uncalibratedScore", "scoreLabel", "scoreCalibrated")
                      if k in result}
        else:
            # No model, or a metadata-only request: the labelled demo engine
            prediction, confidence, frames = echo_verdict(file_name, file_size, file_type)

        identity = model_identity()
        latency_ms = int((time.perf_counter() - started) * 1000)
        engine_used = "live" if (live and media_path) else "simulated"
        log.info("analyzed %s (%s) -> %s %d%% in %d ms", file_name, file_type,
                 prediction, confidence, latency_ms)

        payload = {
            "ok": True,
            "prediction": prediction,
            "confidence": confidence,
            # Compatibility: confidence remains numeric for current clients.
            # Its descriptor is explicit so it cannot be read as a calibrated
            # probability while V5 calibration is still pending.
            "scoreLabel": CFG.SCORE_DESCRIPTION,
            "scoreCalibrated": False,
            # Which engine produced THIS verdict. /api/health also reports
            # the engine, but that is a different request at a different
            # moment — a result read hours later from history must be able
            # to say whether a real model looked at the media or not.
            "engine": engine_used,
            # `riskLevel` is what every page already reads; `risk` and
            # `certainty` are additive. `certainty` is the honest reading of
            # the number — evidence strength, not a probability.
            "riskLevel": inference.risk_for(prediction, confidence),
            "risk": inference.risk_for(prediction, confidence).lower(),
            "certainty": inference.certainty_for(confidence),
            "framesAnalyzed": frames,
            "processingTime": latency_ms,
            "model": identity["name"],
            "device": identity["device"],
            "completedAt": datetime.now(timezone.utc).isoformat(),
            **extras,
        }

        # Recorded off the request thread, after the verdict is decided, and
        # unable to fail it. No media and no filename reach the store.
        #
        # Guarded here as well as inside the store. The store swallows its
        # own errors, but a mistake in *reaching* it is a different failure:
        # while writing this, a missing import turned eleven finished
        # verdicts into 500s. Bookkeeping must never cost a user an answer.
        try:
            store.record_analysis(
                scan_id=scan_id, file_name=file_name,
                file_type=file_type, file_bytes=file_size, result=payload,
                identity=identity, latency_ms=latency_ms, engine=engine_used)
        except Exception:
            log.warning("analytics skipped for this scan", exc_info=True)

        return jsonify(payload)
    finally:
        if owned:
            discard(media_path)


if __name__ == "__main__":
    setup_logging()
    log.info("DeepShield AI backend running at http://localhost:%d", CFG.PORT)
    log.info("config: %s", CFG.summary())

    print(f"DeepShield AI backend running at http://localhost:{CFG.PORT}")
    app.run(host=CFG.HOST, port=CFG.PORT, debug=CFG.DEBUG)
