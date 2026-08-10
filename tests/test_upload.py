"""What the server will and will not take from an upload form.

The rule this suite exists to hold: **an extension is not evidence.** A file
called `photo.png` is a PNG only if its first bytes say so and a decoder
agrees. Every check below is a step of that ladder — size, extension, MIME,
magic bytes, decoder, dimensions, duration — and each one is here because
skipping it hands attacker-controlled bytes to a decoder.
"""
import io
import os

import pytest

pytestmark = pytest.mark.upload


def send(client, blob, filename, endpoint="/api/upload", **fields):
    data = {"file": (io.BytesIO(blob), filename)}
    data.update(fields)
    return client.post(endpoint, data=data, content_type="multipart/form-data")


def send_path(client, path, endpoint="/api/upload", **fields):
    with open(path, "rb") as f:
        return send(client, f.read(), os.path.basename(path), endpoint, **fields)


# ----------------------------------------------------------------- accepted

def test_a_real_image_is_staged(client, fake_face):
    body = send_path(client, fake_face).get_json()
    assert body["ok"] is True
    assert body["uploadId"]


def test_a_staged_upload_exists_on_disk(client, fake_face):
    import app as app_module
    upload_id = send_path(client, fake_face).get_json()["uploadId"]
    assert app_module.staged_upload_path(upload_id), \
        "the id came back but resolves to nothing"


def test_a_real_video_is_staged(client, face_video):
    assert send_path(client, face_video).get_json()["ok"] is True


def test_png_is_accepted(client, png_bytes):
    assert send(client, png_bytes, "square.png").get_json()["ok"] is True


# ----------------------------------------------------------------- refused

def test_missing_file_is_refused(client):
    r = client.post("/api/upload", data={}, content_type="multipart/form-data")
    assert r.status_code == 400
    assert r.get_json()["error_code"] == "NO_FILE"


def test_empty_file_is_refused(client):
    r = send(client, b"", "empty.jpg")
    assert r.status_code == 400
    assert r.get_json()["error_code"] in ("EMPTY_FILE", "BAD_MAGIC", "CORRUPT_MEDIA")


def test_disallowed_extension_is_refused(client):
    r = send(client, b"MZ\x90\x00" + b"\x00" * 2048, "payload.exe")
    assert r.status_code == 400
    assert r.get_json()["error_code"] == "BAD_TYPE"


def test_a_document_is_refused(client):
    doc = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "docs", "MODEL_CARD.md")
    r = send_path(client, doc)
    assert r.status_code == 400


@pytest.mark.parametrize("blob,name,what", [
    (b"MZ\x90\x00" + b"\x00" * 2048, "trojan.png", "a Windows executable"),
    (b"<!DOCTYPE html><html><body>hi</body></html>" * 40, "page.jpg", "an HTML page"),
    (b"PK\x03\x04" + b"\x00" * 2048, "archive.mp4", "a zip archive"),
    (b"%PDF-1.4\n" + b"\x00" * 2048, "doc.png", "a PDF"),
])
def test_a_renamed_file_is_refused(client, blob, name, what):
    """The extension is allowed; the bytes are not."""
    r = send(client, blob, name)
    assert r.status_code == 400, f"{what} got through as {name}"
    assert r.get_json()["error_code"] in ("BAD_MAGIC", "BAD_MIME", "CORRUPT_MEDIA")


def test_a_truncated_image_is_refused(client, png_bytes):
    """Right magic bytes, no usable image behind them — this is the case
    magic-byte checking alone cannot catch, so a decoder has to run."""
    r = send(client, png_bytes[:40], "half.png")
    assert r.status_code == 400
    assert r.get_json()["error_code"] in ("CORRUPT_MEDIA", "IMAGE_TOO_SMALL")


def test_a_corrupt_video_is_refused(client, corrupt_video):
    r = send_path(client, corrupt_video)
    assert r.status_code == 400
    assert r.get_json()["error_code"] == "CORRUPT_MEDIA"


def test_an_empty_video_is_refused(client, empty_video):
    r = send_path(client, empty_video)
    assert r.status_code == 400


def test_an_oversized_upload_is_refused(client):
    from config import CFG
    blob = b"\xff\xd8\xff\xe0" + b"\x00" * (CFG.MAX_UPLOAD_BYTES + 1024)
    r = send(client, blob, "huge.jpg")
    assert r.status_code in (400, 413)


def test_a_decompression_bomb_is_refused(client):
    """A few kilobytes on disk that expand to hundreds of megapixels in
    RAM. The pixel count has to be checked before the decode is trusted."""
    import numpy as np
    from PIL import Image
    from config import CFG

    # Sized to clear OUR limit while staying under Pillow's own bomb guard,
    # so it is this project's check that has to fire and not the library's.
    side = int((CFG.MAX_IMAGE_PIXELS * 1.4) ** 0.5)
    buf = io.BytesIO()
    Image.fromarray(np.zeros((side, side), "uint8")).save(buf, "PNG")
    blob = buf.getvalue()
    assert side * side > CFG.MAX_IMAGE_PIXELS
    assert len(blob) < 5_000_000, "the bomb should be small on disk"

    r = send(client, blob, "bomb.png")
    assert r.status_code == 400
    assert r.get_json()["error_code"] == "IMAGE_TOO_LARGE"


def test_a_bomb_past_pillows_own_guard_is_also_refused(client):
    """Far larger, so Pillow raises before our pixel count is reached. The
    refusal must still be clean rather than a 500."""
    import numpy as np
    from PIL import Image
    from config import CFG

    side = int((CFG.MAX_IMAGE_PIXELS * 4) ** 0.5)
    buf = io.BytesIO()
    Image.fromarray(np.zeros((side, side), "uint8")).save(buf, "PNG")

    r = send(client, buf.getvalue(), "bigger_bomb.png")
    assert r.status_code == 400
    assert r.get_json()["error_code"] in ("IMAGE_TOO_LARGE", "CORRUPT_MEDIA")


# ------------------------------------------------------- analyze endpoint

def test_analyze_refuses_the_same_things(client):
    """`/api/analyze` takes files directly too, and must not be a softer
    door into the same decoder."""
    r = send(client, b"MZ\x90\x00" + b"\x00" * 2048, "trojan.png",
             endpoint="/api/analyze", fileType="image")
    assert r.status_code == 400


def test_an_unknown_upload_id_is_refused(client):
    r = client.post("/api/analyze", json={
        "uploadId": "does-not-exist", "fileName": "x.jpg", "fileType": "image"})
    assert r.status_code in (400, 404)
    assert r.get_json()["ok"] is False


def test_an_analysed_upload_is_deleted(client, engine_ready, fake_face):
    """Media is not kept. Once the verdict is out, the file goes."""
    import app as app_module
    upload_id = send_path(client, fake_face).get_json()["uploadId"]
    assert app_module.staged_upload_path(upload_id)

    client.post("/api/analyze", json={
        "uploadId": upload_id, "fileName": "face.jpeg", "fileType": "image"})
    assert app_module.staged_upload_path(upload_id) is None, \
        "the uploaded file survived its own analysis"
