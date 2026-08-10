"""Regression harness — records what the system does, then proves a
refactor did not change it.

    python scripts/regression_test.py record   # write the baseline
    python scripts/regression_test.py verify   # compare against it

Covers the inference engine directly and every HTTP endpoint. Fields that
legitimately vary between runs (timings, timestamps, ids, the heatmap
image) are compared by shape or ignored — everything else must match
exactly.
"""
import glob
import json
import os
import sys
import urllib.error
import urllib.request
import uuid

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "backend"))
BASELINE = os.path.join(ROOT, "docs", "regression_baseline.json")
API = os.environ.get("DS_TEST_URL", "http://127.0.0.1:5000")

# Volatile: present and of the right type, but never compared by value
VOLATILE = {"processingTime", "completedAt", "uploadId", "heatmapDataUrl"}


# ---------------------------------------------------------------- helpers

def images():
    files = sorted(glob.glob(os.path.join(ROOT, "training", "tpdn_test", "*")))
    return [f for f in files if f.lower().endswith((".jpg", ".jpeg", ".png"))]


def videos():
    """Clips to regression-test the video path against.

    Anything dropped in `training/video_test/` is used. If that folder is
    empty the harness builds its own clip from the sample faces, because
    coverage that only exists when someone remembers to add a file is not
    coverage at all — and on a fresh clone with no clips this suite would
    have quietly checked nothing while still reporting PASS.

    The clip is written once and reused: the encoder is deterministic, so
    the same frames give the same bytes and the same scores.
    """
    folder = os.path.join(ROOT, "training", "video_test")
    found = sorted(glob.glob(os.path.join(folder, "*.mp4")))
    if found:
        return found

    faces = images()[:2]
    if not faces:
        return []
    try:
        import cv2
    except ImportError:
        return []

    os.makedirs(folder, exist_ok=True)
    target = os.path.join(folder, "generated_faces.mp4")
    frames = [cv2.resize(cv2.imread(p), (320, 320)) for p in faces]
    writer = cv2.VideoWriter(target, cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (320, 320))
    for _ in range(3):                      # ~6 seconds, 6 sampled frames
        for frame in frames:
            for _ in range(5):
                writer.write(frame)
    writer.release()
    return [target] if os.path.exists(target) else []


def normalise(obj):
    """Replace volatile values with a type marker so shape is still checked."""
    if isinstance(obj, dict):
        return {k: (f"<{type(v).__name__}>" if k in VOLATILE else normalise(v))
                for k, v in sorted(obj.items())}
    if isinstance(obj, list):
        return [normalise(v) for v in obj]
    if isinstance(obj, float):
        return round(obj, 6)
    return obj


def post_json(path, payload):
    req = urllib.request.Request(
        API + path, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def post_file(path, file_path, fields=None):
    boundary = uuid.uuid4().hex
    parts = []
    for k, v in (fields or {}).items():
        parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode())
    if file_path:
        with open(file_path, "rb") as fh:
            blob = fh.read()
        parts.append(
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; "
            f"filename=\"{os.path.basename(file_path)}\"\r\n"
            f"Content-Type: application/octet-stream\r\n\r\n".encode() + blob + b"\r\n")
    body = b"".join(parts) + f"--{boundary}--\r\n".encode()
    req = urllib.request.Request(
        API + path, data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST")
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


# ---------------------------------------------------------------- suites

def engine_suite():
    """The inference engine on its own — no HTTP involved."""
    import inference
    out = {"engine_available": inference.engine_available(),
           "info": normalise(inference.engine_info())}

    for path in images():
        r = inference.analyze_file(path, "image")
        ex = r.get("explain") or {}
        out["img:" + os.path.basename(path)] = {
            "prediction": r["prediction"],
            "confidence": r["confidence"],
            "framesAnalyzed": r["framesAnalyzed"],
            "disputed": r.get("disputed"),
            "pFake": round(r["ensemble"][0]["pFake"], 4),
            "voters": [v["model"] for v in r["ensemble"]],
            "focusRegion": ex.get("focusRegion"),
            "method": ex.get("method"),
        }

    for path in videos():
        r = inference.analyze_file(path, "video", frame_rate=1.0)
        v = r.get("video") or {}
        out["vid:" + os.path.basename(path)] = {
            "prediction": r["prediction"],
            "confidence": r["confidence"],
            "framesAnalyzed": r["framesAnalyzed"],
            # Aggregation: the three components and what they combined to
            "median": v.get("medianFakeScore"),
            "mean": v.get("meanFakeScore"),
            "topK": v.get("topKFakeScore"),
            "k": v.get("k"),
            "combined": v.get("combinedScore"),
            "peak": v.get("peakFakeScore"),
            "suspicious": v.get("suspiciousFrames"),
            "timestamps": [m["timestamp"] for m in v.get("topTimestamps", [])],
            "timelinePoints": len(v.get("timeline") or []),
            # Temporal signals are descriptive, but a change in them is
            # still a change in behaviour and should have to be explained
            "temporal": v.get("temporal"),
        }
    return out


def api_suite():
    """Every endpoint, including the error paths."""
    out = {}

    with urllib.request.urlopen(API + "/api/health", timeout=30) as r:
        out["GET /api/health"] = {"status": r.status, "body": normalise(json.loads(r.read()))}

    # analyze via staged upload (the path the app actually uses)
    img = images()[0]
    status, body = post_file("/api/upload", img)
    out["POST /api/upload"] = {"status": status, "body": normalise(body)}
    status, body = post_json("/api/analyze",
                             {"uploadId": body.get("uploadId"), "fileName": "sample.jpg",
                              "fileType": "image"})
    out["POST /api/analyze (uploadId)"] = {"status": status, "body": normalise(body)}

    # analyze via multipart in one shot
    status, body = post_file("/api/analyze", img, {"fileType": "image"})
    out["POST /api/analyze (multipart)"] = {"status": status, "body": normalise(body)}

    # A real video through the wire. The engine's video output was already
    # covered above, but nothing checked what survived app.py — which
    # forwards only the keys it names, and dropped the entire `video` block
    # on the way out until this case was added.
    clips = videos()
    if clips:
        status, body = post_file("/api/analyze", clips[0], {"fileType": "video"})
        out["POST /api/analyze (video)"] = {"status": status, "body": normalise(body)}

    # metadata-only request → echo verdict, no file
    status, body = post_json("/api/analyze", {"fileName": "holiday_fake.mp4",
                                              "fileType": "video", "fileSize": 1234})
    out["POST /api/analyze (metadata only)"] = {"status": status, "body": normalise(body)}

    # feedback: happy path and rejection
    status, body = post_json("/api/feedback", {"scanId": "SCAN-TEST", "prediction": "real",
                                               "confidence": 90, "fileType": "image",
                                               "agree": True})
    out["POST /api/feedback"] = {"status": status, "body": normalise(body)}
    status, body = post_json("/api/feedback", {"agree": "nope"})
    out["POST /api/feedback (bad)"] = {"status": status, "body": normalise(body)}

    # upload rejections
    status, body = post_file("/api/upload", None)
    out["POST /api/upload (no file)"] = {"status": status, "body": normalise(body)}
    bad = os.path.join(ROOT, "docs", "MODEL_CARD.md")
    status, body = post_file("/api/upload", bad)
    out["POST /api/upload (bad type)"] = {"status": status, "body": normalise(body)}

    # static routes, including a miss — a missing asset must stay a normal
    # 404 rather than turning into an API-shaped error
    for path in ("/", "/dashboard.html", "/assets/js/utils.js",
                 "/definitely-not-here.html", "/api/not-an-endpoint"):
        try:
            with urllib.request.urlopen(API + path, timeout=30) as r:
                out["GET " + path] = {"status": r.status, "type": r.headers.get_content_type()}
        except urllib.error.HTTPError as e:
            out["GET " + path] = {"status": e.code, "type": e.headers.get_content_type()}
    return out


def collect(with_api):
    data = {"engine": engine_suite()}
    if with_api:
        data["api"] = api_suite()
    return data


# ---------------------------------------------------------------- compare

def diff(old, new, path=""):
    problems = []
    if isinstance(old, dict) and isinstance(new, dict):
        for k in sorted(set(old) | set(new)):
            if k not in old:
                problems.append(f"+ {path}{k} (added: {new[k]!r})")
            elif k not in new:
                problems.append(f"- {path}{k} (removed, was {old[k]!r})")
            else:
                problems += diff(old[k], new[k], f"{path}{k}.")
    elif isinstance(old, list) and isinstance(new, list):
        if len(old) != len(new):
            problems.append(f"~ {path[:-1]} length {len(old)} -> {len(new)}")
        else:
            for i, (a, b) in enumerate(zip(old, new)):
                problems += diff(a, b, f"{path}[{i}].")
    elif old != new:
        problems.append(f"~ {path[:-1]}: {old!r} -> {new!r}")
    return problems


def api_reachable():
    try:
        urllib.request.urlopen(API + "/api/health", timeout=3).read()
        return True
    except Exception:
        return False


def main():
    mode = (sys.argv[1] if len(sys.argv) > 1 else "verify").lower()
    with_api = api_reachable()
    if not with_api:
        print(f"note: {API} is not answering — running engine checks only\n")

    data = collect(with_api)

    if mode == "record":
        os.makedirs(os.path.dirname(BASELINE), exist_ok=True)
        with open(BASELINE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=1, sort_keys=True)
        checks = sum(len(v) for v in data.values())
        print(f"baseline written: {BASELINE}")
        print(f"  {checks} recorded values across {len(data)} suites")
        return 0

    if not os.path.exists(BASELINE):
        sys.exit("no baseline recorded — run: python scripts/regression_test.py record")
    with open(BASELINE, encoding="utf-8") as f:
        old = json.load(f)

    if not with_api and "api" in old:
        old = {k: v for k, v in old.items() if k != "api"}
        print("note: skipping the recorded API suite (server not running)\n")

    problems = diff(old, data)
    if problems:
        print(f"REGRESSION — {len(problems)} difference(s):\n")
        for p in problems:
            print("  " + p)
        return 1
    checks = sum(len(v) for v in data.values())
    print(f"PASS — {checks} values identical to the baseline")
    return 0


if __name__ == "__main__":
    sys.exit(main())
