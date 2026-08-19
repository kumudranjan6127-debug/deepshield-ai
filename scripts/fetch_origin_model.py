"""Install DeepShield's optional full-frame AI-origin ONNX model.

The download is pinned to an immutable Hugging Face revision and verified by
SHA-256 before it replaces anything in ``models/``.  It is intended for setup
or deploy builds; the request path never downloads models.
"""
from __future__ import annotations

import hashlib
import os
import sys
import tempfile
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEST = os.path.join(ROOT, "models", "ai_origin_int8.onnx")
REVISION = "7f067e23521eeb6d6525221af82c613fb746aaff"
URL = (
    "https://huggingface.co/onnx-community/ai-image-detect-distilled-ONNX/"
    f"resolve/{REVISION}/onnx/model_int8.onnx?download=true"
)
SHA256 = "7273cb9cd81e17eae04771010d2199ba6ae34ea2a75a275518c0bc4a2c26ffd2"


def digest(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    os.makedirs(os.path.dirname(DEST), exist_ok=True)
    if os.path.exists(DEST) and digest(DEST) == SHA256:
        print("AI-origin model already installed and verified")
        return 0

    fd, tmp = tempfile.mkstemp(prefix="ai_origin_", suffix=".onnx", dir=os.path.dirname(DEST))
    os.close(fd)
    try:
        print("Downloading pinned AI-origin model…")
        req = urllib.request.Request(URL, headers={"User-Agent": "DeepShield-model-installer/1.0"})
        with urllib.request.urlopen(req, timeout=120) as src, open(tmp, "wb") as dst:
            while True:
                chunk = src.read(1024 * 1024)
                if not chunk:
                    break
                dst.write(chunk)

        got = digest(tmp)
        if got != SHA256:
            print(f"SHA-256 mismatch: expected {SHA256}, got {got}", file=sys.stderr)
            return 2
        os.replace(tmp, DEST)
        print(f"Installed {os.path.relpath(DEST, ROOT)} ({os.path.getsize(DEST)/1e6:.1f} MB)")
        return 0
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
