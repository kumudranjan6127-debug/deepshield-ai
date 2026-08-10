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


def not_a_video():
    return ApiError("URL_NOT_VIDEO", "URL does not point to a video")


def too_large(limit_mb: int):
    return ApiError("TOO_LARGE", f"Video larger than the {limit_mb} MB limit")


def bad_field(field: str, expected: str):
    return ApiError("BAD_FIELD", f"{field} must be {expected}")


def _is_api() -> bool:
    return request.path.startswith("/api/")


def register(app):
    """Attach the handlers. Every API error leaves through here."""

    @app.errorhandler(ApiError)
    def _api_error(e: ApiError):
        log.info("%s %s -> %s: %s", request.method, request.path, e.code, e.message)
        return jsonify(e.payload()), e.status

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
