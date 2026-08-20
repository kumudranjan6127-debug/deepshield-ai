from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UPLOAD_JS = ROOT / "frontend" / "assets" / "js" / "pages" / "upload.js"


def _source():
    return UPLOAD_JS.read_text(encoding="utf-8")


def test_mobile_image_upload_has_large_photo_downscale_path():
    src = _source()
    assert "prepareFileForUpload" in src
    assert "MAX_UPLOAD_IMAGE_PIXELS" in src
    assert "canvas.toBlob" in src
    assert "MAX_UPLOAD_IMAGE_SIDE" in src


def test_upload_surfaces_backend_error_instead_of_masking_it_as_network_failure():
    src = _source()
    assert "const payload = await res.json().catch" in src
    assert "payload.error" in src
    assert "payload.error_code" in src
    assert "IMAGE_TOO_LARGE" in src
    assert "RATE_LIMITED" in src


def test_multipart_filename_is_normalized_for_mobile_photo_picker_files():
    src = _source()
    assert "safeMultipartName" in src
    assert "fd.append('file', uploadFile, safeMultipartName(currentFile, uploadFile))" in src
