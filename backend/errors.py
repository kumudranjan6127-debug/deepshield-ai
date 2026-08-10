"""One error shape, one place to raise from.

Routes raise `ApiError`; `register(app)` turns those — and anything
unexpected — into the same JSON body:

    {"ok": false, "error": "Unsupported file type", "error_code": "BAD_TYPE"}

`error` stays a plain string because that is what the frontend already
reads; `ok` and `error_code` are added on top, so the schema standardises
without breaking a single existing caller.

Only `/api/*` failures are converted. A missing page or asset keeps
Flask's normal HTML 404 — returning JSON there would change how the
browser behaves for static requests.
"""
import logging

from flask import jsonify, request
from werkzeug.exceptions import HTTPException

log = logging.getLogger("deepshield")


class ApiError(Exception):
    """A failure the caller can act on. `code` is stable, `message` is human."""

    def __init__(self, code: str, message: str, status: int = 400):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status

    def payload(self) -> dict:
        return {"ok": False, "error": self.message, "error_code": self.code}


# ---- Named failures, so routes read as intent rather than plumbing ----

def no_file():
    return ApiError("NO_FILE", "No file received")


def bad_type():
    return ApiError("BAD_TYPE", "Unsupported file type")


def upload_not_found():
    """A named uploadId that resolves to nothing.

    This has to be an error rather than a fallback. Staged uploads are
    swept after 30 minutes, so an id going stale is ordinary — and the
    fallback silently produced a demo verdict that was indistinguishable
    from a real one, model name and all."""
    return ApiError("UPLOAD_NOT_FOUND",
                    "That upload has expired or does not exist. Please upload again.",
                    status=404)


def not_a_video():
    return ApiError("URL_NOT_VIDEO", "URL does not point to a video")


def too_large(limit_mb: int):
    return ApiError("TOO_LARGE", f"Video larger than the {limit_mb} MB limit")


def bad_field(field: str, expected: str):
    return ApiError("BAD_FIELD", f"{field} must be {expected}")


# ---- Upload validation ----

def empty_file():
    return ApiError("EMPTY_FILE", "The file is empty")


def bad_mime(mime: str):
    return ApiError("BAD_MIME", f"Unsupported content type: {mime}")


def bad_magic(ext: str):
    # The name says one thing, the bytes say another.
    return ApiError("BAD_MAGIC", f"File contents do not match a {ext.lstrip('.')} file")


def corrupt_media(detail: str = "The file could not be decoded"):
    return ApiError("CORRUPT_MEDIA", detail)


def too_many_pixels(w: int, h: int):
    return ApiError("IMAGE_TOO_LARGE", f"Image is too large to process ({w}×{h})")


def too_small(w: int, h: int):
    return ApiError("IMAGE_TOO_SMALL", f"Image is too small to analyze ({w}×{h})")


def too_long(seconds: float, limit: int):
    return ApiError("VIDEO_TOO_LONG",
                    f"Video is {seconds:.0f}s; the limit is {limit}s")


# ---- URL fetching ----

def insecure_url():
    return ApiError("INSECURE_URL", "Only https:// URLs are accepted")


def blocked_url(reason: str):
    # Deliberately vague to the caller, precise in the log: probing for
    # internal hosts should not get a useful answer back.
    log.warning("blocked url: %s", reason)
    return ApiError("BLOCKED_URL", "That URL cannot be fetched")


# ---- Traffic ----

def rate_limited(retry_after: int):
    e = ApiError("RATE_LIMITED",
                 f"Too many requests — try again in {retry_after}s", 429)
    e.retry_after = retry_after
    return e


def server_busy():
    return ApiError("BUSY", "The server is busy analyzing — please retry shortly", 503)


def _is_api() -> bool:
    return request.path.startswith("/api/")


def register(app):
    """Attach the handlers. Every API error leaves through here."""

    @app.errorhandler(ApiError)
    def _api_error(e: ApiError):
        log.info("%s %s -> %s: %s", request.method, request.path, e.code, e.message)
        response = jsonify(e.payload())
        if getattr(e, "retry_after", None):
            response.headers["Retry-After"] = str(e.retry_after)
        return response, e.status

    @app.errorhandler(ValueError)
    def _value_error(e: ValueError):
        # The engine signals bad input with ValueError (unreadable video,
        # no frames). A 400 is the honest answer, not a 500.
        if not _is_api():
            raise e
        log.info("%s %s -> INVALID_INPUT: %s", request.method, request.path, e)
        return jsonify(ApiError("INVALID_INPUT", str(e)).payload()), 400

    @app.errorhandler(HTTPException)
    def _http_error(e: HTTPException):
        # A missing page or asset must stay a normal 404 — Flask's own
        # response. Only /api/* gets the JSON shape.
        if not _is_api():
            return e
        return jsonify(
            ApiError(f"HTTP_{e.code}", e.description or e.name, e.code).payload()
        ), e.code

    @app.errorhandler(Exception)
    def _unexpected(e: Exception):
        # HTTPException subclasses (404, 405, 413 …) are not failures of
        # ours; hand them back so Flask renders them as it always did.
        if isinstance(e, HTTPException):
            return _http_error(e)
        if not _is_api():
            raise e
        log.exception("%s %s failed", request.method, request.path)
        return jsonify(
            ApiError("INTERNAL", "Something went wrong on the server", 500).payload()
        ), 500
