"""
============================================================
DeepShield AI — Flask backend (Phase 2 skeleton)

Serves the static frontend AND the /api/analyze endpoint.

Phase 2 (now):  echo engine — implements the exact same response
                contract (and the same seeded-verdict logic) as the
                frontend mock in assets/js/api.js, so the full pipe
                can be tested end-to-end before the model exists.
Phase 4 (next): replace verdict_for() with real inference —
                OpenCV preprocessing + MobileNetV3 (PyTorch, CPU).

Run:   venv/Scripts/python backend/app.py   (or: npm run backend)
Open:  http://localhost:5000
============================================================
"""

import os
import time
import uuid
import urllib.request
from datetime import datetime, timezone

from flask import Flask, jsonify, request, send_from_directory

import inference  # real engine (Phase 4) — safe to import before torch/model exist

# Repo layout: backend/app.py → project root is one level up
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = Flask(__name__, static_folder=None)

MODEL_INFO = {
    "name": "MobileNetV3-Small",
    "version": "1.0.0",
    "params": "2.5M",
    "input": "224 × 224",
    "device": "CPU",
    "backend": "PyTorch",
}


# ---------------------------------------------------------- frontend

@app.route("/")
def home():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/<path:path>")
def assets(path):
    """Static catch-all for pages + assets. frontend/ contains only
    public files, and send_from_directory is path-traversal safe —
    backend code, models and uploads are outside the static root."""
    return send_from_directory(FRONTEND_DIR, path)


# ---------------------------------------------------------- helpers

MAX_URL_BYTES = 100 * 1024 * 1024  # 100 MB cap for URL downloads


def risk_for(prediction: str, confidence: int) -> str:
    """Single source of truth for the risk label (echo AND live modes)."""
    if prediction == "deepfake":
        return "High" if confidence >= 85 else "Medium"
    return "Low" if confidence >= 80 else "Medium"


def download_video(url: str, dest: str) -> int:
    """Download a direct MP4 with size/content-type caps. Returns bytes."""
    req = urllib.request.Request(url, headers={"User-Agent": "DeepShield/1.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        ctype = r.headers.get("Content-Type", "")
        if "video" not in ctype and not url.lower().split("?")[0].endswith(".mp4"):
            raise ValueError("URL does not point to a video")
        size = 0
        with open(dest, "wb") as f:
            while True:
                chunk = r.read(256 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > MAX_URL_BYTES:
                    raise ValueError("Video larger than the 100 MB limit")
                f.write(chunk)
    return size


# ---------------------------------------------------------- echo engine (pre-model fallback)

def fnv_hash(s: str) -> int:
    """FNV-1a, bit-identical to DS.util.hash in assets/js/utils.js —
    the mock (JS) and this server produce the SAME verdict for the
    same file, so demos stay consistent across modes."""
    h = 2166136261
    for ch in s:
        h ^= ord(ch)
        h = (h * 16777619) & 0xFFFFFFFF
    return h


def verdict_for(file_name, file_size, file_type):
    """Phase 2 placeholder — port of DS.api._verdictFor.
    Phase 4 replaces this with OpenCV + MobileNetV3 inference."""
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


@app.get("/api/health")
def health():
    live = inference.engine_available()
    return jsonify({
        "status": "ok",
        "engine": "live" if live else "echo",
        "model": MODEL_INFO,
        **inference.engine_info(),
    })


ALLOWED_UPLOAD_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".mp4", ".webm", ".mov"}


@app.post("/api/upload")
def upload():
    """Stage a file for analysis. The browser can't carry a File object
    across page navigation, so the upload page parks it here and passes
    the returned uploadId to processing.html via the scan object."""
    if "file" not in request.files:
        return jsonify({"error": "No file received"}), 400
    f = request.files["file"]
    ext = os.path.splitext(f.filename or "")[1].lower()
    if ext not in ALLOWED_UPLOAD_EXTS:
        return jsonify({"error": "Unsupported file type"}), 400
    upload_id = uuid.uuid4().hex + ext
    f.save(os.path.join(UPLOAD_DIR, upload_id))
    return jsonify({"uploadId": upload_id})


@app.post("/api/analyze")
def analyze():
    started = time.perf_counter()
    live = inference.engine_available()
    tmp_path = None

    try:
        if "file" in request.files:
            # ---- Multipart path: direct uploads
            f = request.files["file"]
            file_name = f.filename or "upload"
            file_type = request.form.get("fileType", "image")
            frame_rate = float(request.form.get("frameRate", 1))
            if live:
                ext = os.path.splitext(file_name)[1] or (".mp4" if file_type == "video" else ".jpg")
                tmp_path = os.path.join(UPLOAD_DIR, uuid.uuid4().hex + ext)
                f.save(tmp_path)
                file_size = os.path.getsize(tmp_path)
            else:
                file_size = len(f.read())
        else:
            # ---- JSON path: MP4-URL scans (and the pre-Phase-4 live stub)
            data = request.get_json(silent=True) or {}
            url = data.get("url", "")
            file_name = data.get("fileName") or (url.rstrip("/").rsplit("/", 1)[-1] if url else "video.mp4")
            file_size = data.get("fileSize")
            file_type = data.get("fileType", "video")
            frame_rate = float(data.get("frameRate", 1))
            upload_id = data.get("uploadId")
            if live and upload_id:
                # File was staged earlier via /api/upload (basename → no traversal)
                staged = os.path.join(UPLOAD_DIR, os.path.basename(upload_id))
                if os.path.exists(staged):
                    tmp_path = staged
                    file_size = file_size or os.path.getsize(staged)
            elif live and url:
                tmp_path = os.path.join(UPLOAD_DIR, uuid.uuid4().hex + ".mp4")
                file_size = download_video(url, tmp_path)

        extras = {}
        if live and tmp_path:
            # ---- REAL inference: OpenCV sampling + MobileNetV3 on CPU
            result = inference.analyze_file(tmp_path, file_type, frame_rate)
            prediction = result["prediction"]
            confidence = result["confidence"]
            frames = result["framesAnalyzed"]
            # explainability payload for the result/report pages
            extras = {k: result.get(k) for k in ("ensemble", "disputed", "explain")}
        else:
            # ---- Echo fallback (no model yet, or metadata-only request)
            prediction, confidence, frames = verdict_for(file_name, file_size, file_type)

        return jsonify({
            "prediction": prediction,
            "confidence": confidence,
            "riskLevel": risk_for(prediction, confidence),
            "framesAnalyzed": frames,
            "processingTime": int((time.perf_counter() - started) * 1000),
            "model": MODEL_INFO["name"],
            "device": MODEL_INFO["device"],
            "completedAt": datetime.now(timezone.utc).isoformat(),
            **extras,
        })

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    finally:
        # Uploads are temporary by design — always clean up
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


if __name__ == "__main__":
    print("DeepShield AI backend running at http://localhost:5000")
    app.run(host="127.0.0.1", port=5000, debug=True)
