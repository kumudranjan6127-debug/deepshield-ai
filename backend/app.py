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
import time
import urllib.request
import uuid
from datetime import datetime, timezone

from flask import Flask, jsonify, request, send_from_directory

import errors
import inference
import security
from config import CFG

log = logging.getLogger("deepshield")

# Traffic controls, shared by every request
limiter = security.RateLimiter(CFG.RATE_LIMIT, CFG.RATE_WINDOW_SECONDS)
gate = security.InferenceGate(CFG.MAX_CONCURRENT_ANALYSES, CFG.QUEUE_WAIT_SECONDS)


def client_key() -> str:
    """Who is calling. X-Forwarded-For is only consulted behind a proxy we
    were told to trust — otherwise any client could spoof its identity and
    walk around the rate limit."""
    if CFG.TRUST_PROXY:
        forwarded = request.headers.get("X-Forwarded-For", "")
        if forwarded:
            return forwarded.split(",")[0].strip()
    return request.remote_addr or "unknown"


def setup_logging():
    """One configuration for the whole process; modules just get a logger."""
    logging.basicConfig(
        level=getattr(logging, CFG.LOG_LEVEL, logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("werkzeug").setLevel(logging.WARNING)  # per-request noise


CFG.ensure_dirs()
app = Flask(__name__, static_folder=None)
# Refused by Werkzeug before the body is read into memory.
app.config["MAX_CONTENT_LENGTH"] = CFG.MAX_UPLOAD_BYTES
errors.register(app)


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

def model_identity() -> dict:
    """What is actually loaded, asked of the engine rather than hardcoded.

    With no model present the fields say so rather than describing a model
    that is not running — the UI shows "Simulated (demo)" in that state."""
    info = inference.engine_info()
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


def download_video(url: str, dest: str) -> int:
    """Fetch a direct video with size and content-type caps. Returns bytes."""
    req = urllib.request.Request(url, headers={"User-Agent": "DeepShield/1.0"})
    with urllib.request.urlopen(req, timeout=CFG.URL_TIMEOUT_SECONDS) as r:
        ctype = r.headers.get("Content-Type", "")
        if "video" not in ctype and not url.lower().split("?")[0].endswith(".mp4"):
            raise errors.not_a_video()
        size = 0
        with open(dest, "wb") as f:
            while True:
                chunk = r.read(256 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > CFG.MAX_URL_BYTES:
                    raise errors.too_large(CFG.MAX_URL_BYTES // (1024 * 1024))
                f.write(chunk)
    return size


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
    return jsonify({
        "ok": True,
        "status": "ok",
        "engine": "live" if live else "echo",
        "model": model_identity(),
        # The frontend labels confidence numbers with these, rather than
        # keeping its own copy of the thresholds.
        "certainty_bands": inference.certainty_bands(),
        "calibrated": False,   # no reliability curve has ever been measured
        **inference.engine_info(),
    })


@app.post("/api/feedback")
def feedback():
    """Record whether the user agreed with a verdict.

    Stores no media and nothing personal — the verdict and a thumbs
    up/down. This is an evaluation signal, never a training label:
    nothing here reaches the model automatically."""
    d = request.get_json(silent=True) or {}
    if not isinstance(d.get("agree"), bool):
        raise errors.bad_field("agree", "true or false")

    entry = {
        "at": datetime.now(timezone.utc).isoformat(),
        "scanId": str(d.get("scanId", ""))[:40],
        "prediction": str(d.get("prediction", ""))[:20],
        "confidence": d.get("confidence"),
        "fileType": str(d.get("fileType", ""))[:10],
        "agree": d["agree"],
    }
    with open(CFG.FEEDBACK_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
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
        staged = staged_upload_path(d.get("uploadId"))
        if staged:
            return (file_name, file_type, file_size or os.path.getsize(staged),
                    frame_rate, staged, True)
        if url:
            path = new_temp_path(".mp4")
            size = security.safe_download(url, path)   # scheme + DNS + IP + redirects
            try:
                security.validate_media_file(path, "video")
            except Exception:
                discard(path)
                raise
            return file_name, "video", size, frame_rate, path, True

    return file_name, file_type, file_size, frame_rate, None, False


@app.post("/api/analyze")
def analyze():
    started = time.perf_counter()
    limiter.check(client_key())
    live = inference.engine_available()
    media_path = owned = None

    try:
        (file_name, file_type, file_size,
         frame_rate, media_path, owned) = _read_request(live)

        extras = {}
        if live and media_path:
            # One analysis per worker slot; the rest wait, then get a 503
            with gate:
                result = inference.analyze_file(media_path, file_type, frame_rate)
            prediction = result["prediction"]
            confidence = result["confidence"]
            frames = result["framesAnalyzed"]
            extras = {k: result.get(k) for k in ("ensemble", "disputed", "explain")}
        else:
            # No model, or a metadata-only request: the labelled demo engine
            prediction, confidence, frames = echo_verdict(file_name, file_size, file_type)

        identity = model_identity()
        log.info("analyzed %s (%s) -> %s %d%% in %d ms", file_name, file_type,
                 prediction, confidence, int((time.perf_counter() - started) * 1000))

        return jsonify({
            "ok": True,
            "prediction": prediction,
            "confidence": confidence,
            # `riskLevel` is what every page already reads; `risk` and
            # `certainty` are additive. `certainty` is the honest reading of
            # the number — evidence strength, not a probability.
            "riskLevel": inference.risk_for(prediction, confidence),
            "risk": inference.risk_for(prediction, confidence).lower(),
            "certainty": inference.certainty_for(confidence),
            "framesAnalyzed": frames,
            "processingTime": int((time.perf_counter() - started) * 1000),
            "model": identity["name"],
            "device": identity["device"],
            "completedAt": datetime.now(timezone.utc).isoformat(),
            **extras,
        })
    finally:
        if owned:
            discard(media_path)


if __name__ == "__main__":
    setup_logging()
    log.info("DeepShield AI backend running at http://localhost:%d", CFG.PORT)
    log.info("config: %s", CFG.summary())

    # Anything left staged from a previous run is already abandoned
    security.cleanup_uploads()
    security.start_cleanup_thread()

    print(f"DeepShield AI backend running at http://localhost:{CFG.PORT}")
    app.run(host=CFG.HOST, port=CFG.PORT, debug=CFG.DEBUG)
