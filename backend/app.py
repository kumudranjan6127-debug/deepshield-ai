"""DeepShield AI Flask backend.

The application layer deliberately stays thin:

    Frontend -> API -> validation/security -> inference -> response

Model facts come from inference.py, limits from config.py, and upload/URL
security from security.py.  The API forwards detector provenance instead of
collapsing an AI-generated/inconclusive result back into a generic real/fake
label.
"""
from __future__ import annotations

import json
import logging
import math
import os
import time
import uuid
from datetime import datetime, timezone

from flask import Flask, g, jsonify, redirect, request, send_from_directory

import errors
import inference
import security
import store
from config import CFG

log = logging.getLogger("deepshield")
limiter = security.RateLimiter(CFG.RATE_LIMIT, CFG.RATE_WINDOW_SECONDS)
gate = security.InferenceGate(CFG.MAX_CONCURRENT_ANALYSES, CFG.QUEUE_WAIT_SECONDS)


def client_key() -> str:
    if CFG.TRUST_PROXY:
        forwarded = request.headers.get("X-Forwarded-For", "")
        if forwarded:
            return forwarded.split(",")[0].strip()
    return request.remote_addr or "unknown"


class JsonLines(logging.Formatter):
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
    level = getattr(logging, CFG.LOG_LEVEL, logging.INFO)
    formatter = (JsonLines() if CFG.LOG_JSON else logging.Formatter(
        "%(asctime)s %(levelname)-7s %(name)s  %(message)s", datefmt="%H:%M:%S"))
    handlers = [logging.StreamHandler()]
    if CFG.LOG_FILE:
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
    logging.getLogger("werkzeug").setLevel(logging.WARNING)


CFG.ensure_dirs()
app = Flask(__name__, static_folder=None)
app.config["MAX_CONTENT_LENGTH"] = CFG.MAX_UPLOAD_BYTES
errors.register(app)


def request_is_secure() -> bool:
    if request.is_secure:
        return True
    if CFG.TRUST_PROXY:
        return request.headers.get("X-Forwarded-Proto", "").split(",")[0].strip() == "https"
    return False


@app.before_request
def tag_request():
    g.request_id = uuid.uuid4().hex[:12]
    g.started = time.perf_counter()


@app.after_request
def log_api_call(response):
    if request.path.startswith("/api/") and request.path != "/api/health":
        elapsed = int((time.perf_counter() - getattr(g, "started", time.perf_counter())) * 1000)
        log.info("%s %s -> %s in %d ms [%s]", request.method, request.path,
                 response.status_code, elapsed, getattr(g, "request_id", "-"))
    return response


@app.before_request
def enforce_https():
    if not CFG.FORCE_HTTPS or request_is_secure():
        return None
    if request.method not in ("GET", "HEAD"):
        raise errors.insecure_request()
    return redirect(request.url.replace("http://", "https://", 1), code=308)


@app.before_request
def cors_preflight():
    if request.method != "OPTIONS":
        return None
    allowed = security.cors_headers(request.headers.get("Origin", ""))
    response = app.make_response(("", 204 if allowed else 403))
    response.headers.extend(allowed)
    return response


@app.after_request
def harden(response):
    for name, value in security.security_headers(request_is_secure()).items():
        response.headers.setdefault(name, value)
    for name, value in security.cors_headers(request.headers.get("Origin", "")).items():
        response.headers[name] = value
    return response


@app.route("/")
def home():
    return send_from_directory(CFG.FRONTEND_DIR, "landing.html")


@app.route("/<path:path>")
def assets(path):
    return send_from_directory(CFG.FRONTEND_DIR, path)


def model_identity() -> dict:
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
        "model_name": info.get("model_name", "DeepShield"),
        "architecture": info.get("architecture"),
        "version": info.get("version"),
        "runtime": info.get("runtime"),
        "input_size": size,
        "name": info.get("architecture"),
        "params": info.get("params") or "—",
        "input": f"{size} × {size}" if size else "—",
        "backend": info.get("runtime", "none"),
        "device": "CPU",
    }


def staged_upload_path(upload_id: str) -> str | None:
    if not upload_id:
        return None
    path = os.path.join(CFG.UPLOAD_DIR, os.path.basename(str(upload_id)))
    return path if os.path.isfile(path) else None


def new_temp_path(suffix: str) -> str:
    return os.path.join(CFG.UPLOAD_DIR, uuid.uuid4().hex + suffix)


def discard(path: str | None):
    if not path or not os.path.exists(path):
        return
    try:
        os.remove(path)
    except OSError as exc:
        log.warning("could not remove temporary upload: %s", exc)


def _frame_rate(value) -> float:
    try:
        rate = float(value)
    except (TypeError, ValueError):
        raise errors.bad_field("frameRate", "a number between 0.25 and 10")
    if not math.isfinite(rate) or not 0.25 <= rate <= 10.0:
        raise errors.bad_field("frameRate", "a number between 0.25 and 10")
    return rate


def _kind_from_path(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext in security.IMAGE_EXTS:
        return "image"
    if ext in security.VIDEO_EXTS:
        return "video"
    raise errors.bad_type()


def fnv_hash(value: str) -> int:
    h = 2166136261
    for ch in value:
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
    confidence = 72 + (seed % 26)
    frames = 24 + (seed % 37) if file_type == "video" else 1
    return prediction, confidence, frames


@app.get("/api/health")
def health():
    live = inference.engine_available()
    info = inference.engine_info()
    return jsonify({
        "ok": True,
        "status": "healthy",
        "engine": "live" if live else "echo",
        "model": model_identity(),
        "certainty_bands": inference.certainty_bands(),
        "calibrated": bool(info.get("calibrated", False)),
        **info,
    })


@app.get("/api/version")
def version():
    identity = model_identity()
    info = inference.engine_info()
    live = inference.engine_available()
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
        "calibrated": bool(info.get("calibrated", False)),
        "origin_detector": info.get("origin_detector"),
    })


@app.post("/api/feedback")
def feedback():
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
    limiter.check(client_key())
    if "file" not in request.files:
        raise errors.no_file()
    uploaded = request.files["file"]
    ext, kind = security.validate_upload(uploaded)
    upload_id = uuid.uuid4().hex + ext
    path = os.path.join(CFG.UPLOAD_DIR, upload_id)
    uploaded.save(path)
    try:
        details = security.validate_media_file(path, kind)
    except Exception:
        discard(path)
        raise
    log.info("staged %s upload (%s)", kind, details)
    return jsonify({"ok": True, "uploadId": upload_id})


def _read_request(live: bool):
    """Return name, kind, bytes, frame rate, local path, and ownership flag."""
    if "file" in request.files:
        uploaded = request.files["file"]
        file_name = uploaded.filename or "upload"
        frame_rate = _frame_rate(request.form.get("frameRate", CFG.DEFAULT_FRAME_RATE))
        ext, kind = security.validate_upload(uploaded)
        declared = request.form.get("fileType", kind)
        if declared != kind:
            log.info("declared media type differed from decoded container; trusting file")
        if live:
            path = new_temp_path(ext)
            uploaded.save(path)
            try:
                security.validate_media_file(path, kind)
            except Exception:
                discard(path)
                raise
            return file_name, kind, os.path.getsize(path), frame_rate, path, True
        return file_name, kind, len(uploaded.read()), frame_rate, None, False

    data = request.get_json(silent=True) or {}
    url = str(data.get("url") or "")
    file_name = str(data.get("fileName") or (
        url.rstrip("/").rsplit("/", 1)[-1] if url else "video.mp4"))
    # Historic safe fallback: exactly "video" uses the video pipeline;
    # everything else is treated as image rather than dispatched on arbitrary
    # user-controlled strings. A staged file ignores this and trusts its
    # already-validated container extension instead.
    declared = "video" if str(data.get("fileType", "image")) == "video" else "image"
    file_size = data.get("fileSize")
    frame_rate = _frame_rate(data.get("frameRate", CFG.DEFAULT_FRAME_RATE))

    if live:
        upload_id = data.get("uploadId")
        staged = staged_upload_path(upload_id)
        if staged:
            kind = _kind_from_path(staged)
            return (file_name, kind, file_size or os.path.getsize(staged),
                    frame_rate, staged, True)
        if upload_id:
            raise errors.upload_not_found()
        if url:
            path = new_temp_path(".mp4")
            try:
                size = security.safe_download(url, path)
                security.validate_media_file(path, "video")
            except Exception:
                discard(path)
                raise
            return file_name, "video", size, frame_rate, path, True

    return file_name, declared, file_size, frame_rate, None, False


_RESULT_FIELDS = (
    "ensemble", "disputed", "explain", "video", "faceFound", "facesFound",
    "findingType", "insufficientEvidence", "reason", "combiner",
)


@app.post("/api/analyze")
def analyze():
    started = time.perf_counter()
    limiter.check(client_key())
    live = inference.engine_available()
    media_path = None
    owned = False

    incoming_json = request.get_json(silent=True) or {}
    scan_id = str(incoming_json.get("scanId") or request.form.get("scanId") or "")[:64]

    try:
        (file_name, file_type, file_size,
         frame_rate, media_path, owned) = _read_request(live)

        extras = {}
        if live and media_path:
            with gate:
                result = inference.analyze_file(media_path, file_type, frame_rate)
            prediction = result["prediction"]
            confidence = int(result["confidence"])
            frames = int(result["framesAnalyzed"])
            extras = {key: result[key] for key in _RESULT_FIELDS if key in result}
        else:
            prediction, confidence, frames = echo_verdict(file_name, file_size, file_type)

        identity = model_identity()
        latency_ms = int((time.perf_counter() - started) * 1000)
        engine_used = "live" if (live and media_path) else "simulated"
        log.info("analyzed %s media -> %s %d%% in %d ms",
                 file_type, prediction, confidence, latency_ms)

        payload = {
            "ok": True,
            "prediction": prediction,
            "confidence": confidence,
            "engine": engine_used,
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
    security.cleanup_uploads()
    security.start_cleanup_thread()
    print(f"DeepShield AI backend running at http://localhost:{CFG.PORT}")
    app.run(host=CFG.HOST, port=CFG.PORT, debug=CFG.DEBUG)
