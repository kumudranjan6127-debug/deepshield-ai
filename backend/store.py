"""Privacy-minimal analytics and feedback persistence.

No media, filenames, IP addresses or user identity are stored.  DS_ANALYTICS=0
is a hard kill switch: when disabled, neither analyses nor feedback are
written anywhere, including the local JSONL fallback.
"""
import json
import logging
import os
import threading
from datetime import datetime, timezone

from config import CFG

log = logging.getLogger("deepshield")
__all__ = ["record_analysis", "record_feedback", "summary", "backend_name",
           "ensure_schema"]
_schema_ready = False
_lock = threading.Lock()


def _psycopg():
    try:
        import psycopg
        return psycopg
    except ImportError:
        return None


def enabled() -> bool:
    return bool(CFG.ANALYTICS and CFG.DATABASE_URL and _psycopg())


def backend_name() -> str:
    if not CFG.ANALYTICS:
        return "off"
    if not CFG.DATABASE_URL:
        return "file"
    return "postgres" if _psycopg() else "file (psycopg not installed)"


def _connect():
    psycopg = _psycopg()
    return psycopg.connect(CFG.DATABASE_URL,
                           connect_timeout=CFG.DB_TIMEOUT_SECONDS)


SCHEMA = """
CREATE TABLE IF NOT EXISTS analyses (
    id BIGSERIAL PRIMARY KEY, at TIMESTAMPTZ NOT NULL DEFAULT now(),
    scan_id TEXT, file_type TEXT, file_ext TEXT, file_bytes BIGINT,
    prediction TEXT, confidence INTEGER, certainty TEXT, risk TEXT,
    engine TEXT, model_version TEXT, runtime TEXT, frames INTEGER,
    suspicious INTEGER, faces INTEGER, face_found BOOLEAN, latency_ms INTEGER
);
CREATE INDEX IF NOT EXISTS analyses_scan_id ON analyses (scan_id);
CREATE INDEX IF NOT EXISTS analyses_at ON analyses (at DESC);
CREATE TABLE IF NOT EXISTS feedback (
    id BIGSERIAL PRIMARY KEY, at TIMESTAMPTZ NOT NULL DEFAULT now(),
    scan_id TEXT, prediction TEXT, confidence INTEGER, file_type TEXT,
    agree BOOLEAN NOT NULL, note TEXT
);
CREATE INDEX IF NOT EXISTS feedback_scan_id ON feedback (scan_id);
CREATE INDEX IF NOT EXISTS feedback_agree ON feedback (agree);
ALTER TABLE analyses ADD COLUMN IF NOT EXISTS faces INTEGER;
ALTER TABLE analyses ADD COLUMN IF NOT EXISTS face_found BOOLEAN;
"""


def ensure_schema(force=False):
    global _schema_ready
    if not enabled():
        return False
    with _lock:
        if _schema_ready and not force:
            return True
        try:
            with _connect() as conn, conn.cursor() as cur:
                cur.execute(SCHEMA)
                conn.commit()
            _schema_ready = True
            log.info("analytics schema ready (postgres)")
            return True
        except Exception as exc:
            log.warning("analytics schema unavailable: %s: %s",
                        type(exc).__name__, exc)
            return False


def _insert(table, row):
    try:
        if not ensure_schema():
            return
        columns = ", ".join(row)
        holders = ", ".join(["%s"] * len(row))
        with _connect() as conn, conn.cursor() as cur:
            cur.execute(f"INSERT INTO {table} ({columns}) VALUES ({holders})",
                        list(row.values()))
            conn.commit()
    except Exception as exc:
        log.warning("analytics write failed (%s): %s: %s",
                    table, type(exc).__name__, exc)


def _write_async(table, row):
    threading.Thread(target=_insert, args=(table, row), daemon=True).start()


def _append_jsonl(path, entry):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as exc:
        log.warning("could not append %s: %s", path, exc)


def _extension_of(file_name):
    ext = os.path.splitext(str(file_name or ""))[1].lower()
    return ext[:8] if ext.startswith(".") else ""


def record_analysis(*, scan_id, file_name, file_type, file_bytes, result,
                    identity, latency_ms, engine):
    if not CFG.ANALYTICS:
        return
    video = result.get("video") or {}
    row = {
        "scan_id": str(scan_id or "")[:64],
        "file_type": str(file_type or "")[:10],
        "file_ext": _extension_of(file_name),
        "file_bytes": int(file_bytes or 0),
        "prediction": str(result.get("prediction") or "")[:20],
        "confidence": int(result.get("confidence") or 0),
        "certainty": str(result.get("certainty") or "")[:20],
        "risk": str(result.get("risk") or "")[:10],
        "engine": str(engine or "")[:12],
        "model_version": str(identity.get("version") or "")[:32],
        "runtime": str(identity.get("runtime") or "")[:16],
        "frames": int(result.get("framesAnalyzed") or 0),
        "suspicious": int(video.get("suspiciousFrames") or 0),
        "faces": int(result["facesFound"]) if "facesFound" in result else -1,
        "face_found": bool(result["faceFound"]) if "faceFound" in result else None,
        "latency_ms": int(latency_ms or 0),
    }
    if enabled():
        _write_async("analyses", row)


def record_feedback(entry):
    """Persist evaluation feedback only when analytics is explicitly enabled."""
    if not CFG.ANALYTICS:
        return

    # Without a database, an enabled local/dev instance retains the old JSONL
    # behaviour.  The global off switch above means privacy mode truly writes
    # nothing at all.
    _append_jsonl(CFG.FEEDBACK_PATH, {**entry, "at": _now()})
    if enabled():
        _write_async("feedback", {
            "scan_id": str(entry.get("scanId") or "")[:64],
            "prediction": str(entry.get("prediction") or "")[:20],
            "confidence": int(entry.get("confidence") or 0),
            "file_type": str(entry.get("fileType") or "")[:10],
            "agree": bool(entry.get("agree")),
            "note": str(entry.get("note") or "")[:280] or None,
        })


def _now():
    return datetime.now(timezone.utc).isoformat()


def summary(limit_days=30):
    if not enabled():
        return {"backend": backend_name(), "available": False}
    queries = {
        "analyses": "SELECT count(*) FROM analyses WHERE at > now() - %s::interval",
        "feedback": "SELECT count(*) FROM feedback WHERE at > now() - %s::interval",
        "disagreements": ("SELECT count(*) FROM feedback "
                          "WHERE agree = false AND at > now() - %s::interval"),
    }
    window = f"{int(limit_days)} days"
    out = {"backend": backend_name(), "available": True, "window_days": limit_days}
    try:
        with _connect() as conn, conn.cursor() as cur:
            for key, sql in queries.items():
                cur.execute(sql, (window,))
                out[key] = int(cur.fetchone()[0])
            cur.execute(
                "SELECT prediction, count(*) FROM analyses "
                "WHERE at > now() - %s::interval GROUP BY prediction", (window,))
            out["by_prediction"] = {r[0] or "?": int(r[1]) for r in cur.fetchall()}
            cur.execute(
                "SELECT file_type, count(*) FROM analyses "
                "WHERE at > now() - %s::interval GROUP BY file_type", (window,))
            out["by_file_type"] = {r[0] or "?": int(r[1]) for r in cur.fetchall()}
            cur.execute(
                "SELECT round(avg(latency_ms)), max(latency_ms) FROM analyses "
                "WHERE at > now() - %s::interval", (window,))
            row = cur.fetchone()
            out["latency_ms"] = {"mean": int(row[0] or 0), "max": int(row[1] or 0)}
            cur.execute(
                "SELECT f.scan_id, f.prediction, f.confidence, f.file_type, f.at "
                "FROM feedback f WHERE f.agree = false ORDER BY f.at DESC LIMIT 20")
            out["recent_disagreements"] = [
                {"scan_id": r[0], "prediction": r[1], "confidence": r[2],
                 "file_type": r[3], "at": r[4].isoformat() if r[4] else None}
                for r in cur.fetchall()]
        if out["feedback"]:
            out["disagreement_rate"] = round(
                out["disagreements"] / out["feedback"], 4)
    except Exception as exc:
        log.warning("analytics summary failed: %s: %s", type(exc).__name__, exc)
        return {"backend": backend_name(), "available": False,
                "error": type(exc).__name__}
    return out
