"""Every path, limit and environment switch in one place.

Nothing here imports the rest of the app, so it can be read (and tested)
on its own. Import it as `from config import CFG`.
"""
import os


def _bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "on", "yes")


def _int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _str(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


class Config:
    """Resolved once at import. Read-only in practice."""

    # ---- Layout: backend/config.py → the project root is one level up ----
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
    MODELS_DIR = os.path.join(BASE_DIR, "models")
    UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
    DATA_DIR = os.path.join(BASE_DIR, "data")
    FEEDBACK_PATH = os.path.join(DATA_DIR, "feedback.jsonl")

    # ---- Model files ----
    CKPT_PATH = os.path.join(MODELS_DIR, "deepshield_mobilenetv3.pth")
    ONNX_PATH = os.path.join(MODELS_DIR, "deepshield.onnx")
    ONNX_META_PATH = ONNX_PATH + ".json"
    YUNET_PATH = os.path.join(MODELS_DIR, "face_detection_yunet.onnx")

    # ---- Server ----
    PORT = _int("PORT", 5000)
    HOST = _str("DS_HOST", "127.0.0.1")
    # Debug off by default: its reloader runs a second process that respawns
    # the first, which is how a dozen instances once piled up.
    DEBUG = _bool("DS_DEBUG")
    LOG_LEVEL = _str("DS_LOG_LEVEL", "INFO").upper()

    # ---- Engine ----
    # DS_ENGINE=echo forces the openly-labelled simulated mode.
    FORCE_ECHO = _str("DS_ENGINE").lower() == "echo"
    VERIFIERS = _bool("DS_VERIFIERS")

    # ---- Upload / analysis limits ----
    MAX_URL_BYTES = _int("DS_MAX_URL_MB", 100) * 1024 * 1024
    URL_TIMEOUT_SECONDS = _int("DS_URL_TIMEOUT", 30)
    ALLOWED_UPLOAD_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".mp4", ".webm", ".mov"}
    MAX_VIDEO_FRAMES = _int("DS_MAX_FRAMES", 60)
    DEFAULT_FRAME_RATE = 1.0

    # ---- Security: uploads ----
    # Flask rejects anything larger before the body is buffered.
    MAX_UPLOAD_BYTES = _int("DS_MAX_UPLOAD_MB", 100) * 1024 * 1024
    MAX_IMAGE_PIXELS = _int("DS_MAX_IMAGE_PIXELS", 40_000_000)  # ~6300² — bomb guard
    MAX_VIDEO_SECONDS = _int("DS_MAX_VIDEO_SECONDS", 300)

    # ---- Security: outbound URLs (SSRF) ----
    # Plain HTTP is off by default; set DS_ALLOW_HTTP=1 for local testing.
    ALLOW_HTTP_URLS = _bool("DS_ALLOW_HTTP")
    MAX_REDIRECTS = _int("DS_MAX_REDIRECTS", 3)

    # ---- Security: traffic ----
    # Only set behind a proxy you control; otherwise clients can forge
    # X-Forwarded-For and each request looks like a new IP.
    TRUST_PROXY = _bool("DS_TRUST_PROXY")
    RATE_LIMIT = _int("DS_RATE_LIMIT", 5)              # requests per client
    RATE_WINDOW_SECONDS = _int("DS_RATE_WINDOW", 60)   # per this window
    MAX_CONCURRENT_ANALYSES = _int("DS_WORKERS", 2)    # CPU-bound, ~200 MB each
    QUEUE_WAIT_SECONDS = _int("DS_QUEUE_WAIT", 20)

    # ---- Security: housekeeping ----
    UPLOAD_TTL_SECONDS = _int("DS_UPLOAD_TTL", 30 * 60)
    CLEANUP_INTERVAL_SECONDS = _int("DS_CLEANUP_INTERVAL", 5 * 60)

    # ---- Inference tuning ----
    MAX_IMAGE_SIDE = _int("DS_MAX_IMAGE_SIDE", 1024)   # oversized inputs are unlike training data
    JPEG_NORMALISE_QUALITY = _int("DS_JPEG_QUALITY", 88)
    OCCLUSION_GRID = _int("DS_OCCLUSION_GRID", 6)
    MAX_FORWARD_BATCH = _int("DS_MAX_BATCH", 8)        # OpenCV's DNN engine dies on large batches
    OWN_WEIGHT = 0.75                                  # verifiers share the remainder
    VERIFIER_OVERRULE_AT = 0.85                        # and only when they all agree

    # ---- How a confidence number is allowed to be described ----
    # A percentage invites the reading "94% probability this is fake". It is
    # not that: it is how strongly this model separated this image, and the
    # model has never been calibrated, so the number carries no probabilistic
    # promise. These bands turn it into a claim about *evidence strength*,
    # which is what it can honestly support.
    #
    # PROVISIONAL. The cut points below are the ones specified for Phase 5,
    # not ones derived from data — nothing has been measured that would
    # justify 70 over 75. `scripts/evaluate.py` prints the observed accuracy
    # and occupancy of each band; replace these with the measured values as
    # soon as there is a labelled set to measure on. Until then, treat the
    # boundaries as a vocabulary, not a finding.
    #
    # (lower bound inclusive, key, human label)
    CERTAINTY_BANDS = (
        (90, "very_strong", "Very strong evidence"),
        (70, "strong",      "Strong evidence"),
        (30, "uncertain",   "Uncertain"),
        (0,  "low_evidence", "Low evidence"),
    )

    @classmethod
    def ensure_dirs(cls):
        os.makedirs(cls.UPLOAD_DIR, exist_ok=True)
        os.makedirs(cls.DATA_DIR, exist_ok=True)

    @classmethod
    def summary(cls) -> dict:
        """What the process is actually running with — logged at startup."""
        return {
            "port": cls.PORT,
            "debug": cls.DEBUG,
            "force_echo": cls.FORCE_ECHO,
            "verifiers": cls.VERIFIERS,
            "max_upload_mb": cls.MAX_UPLOAD_BYTES // (1024 * 1024),
            "max_url_mb": cls.MAX_URL_BYTES // (1024 * 1024),
            "max_video_seconds": cls.MAX_VIDEO_SECONDS,
            "rate_limit": f"{cls.RATE_LIMIT}/{cls.RATE_WINDOW_SECONDS}s",
            "workers": cls.MAX_CONCURRENT_ANALYSES,
            "allow_http_urls": cls.ALLOW_HTTP_URLS,
            "upload_ttl_min": cls.UPLOAD_TTL_SECONDS // 60,
        }


CFG = Config()
