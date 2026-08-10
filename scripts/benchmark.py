"""Where the time actually goes.

    python scripts/benchmark.py              # the standard matrix
    python scripts/benchmark.py --repeats 5
    python scripts/benchmark.py --only image-1024

Measures the pipeline stage by stage so optimisation has something to aim
at. The engine is instrumented by wrapping its methods here rather than by
adding timers to `inference.py` — production pays nothing for a
measurement that only this script needs.

Reported per case:

    staging       validate the upload and write it to disk
    decode+detect open the media and find a face (YuNet)
    normalise     the JPEG round trip that puts inputs in one domain
    prepare       resize to 224 and normalise to a tensor
    forward       the model itself
    explain       occlusion sensitivity (images only, 36 extra forwards)
    aggregate     combining frame scores (video only)

RSS is sampled by a background thread, because the peak of a 60-frame video
sits in the middle of the run and a before/after delta would miss it.

**The clip profile matters, and this one is conservative.** The generated
clips are 480x480 at 10 fps, so sampling at 1 fps discards 9 frames in 10.
A phone records 720p at 30 fps and discards 29 in 30, where both the decode
and the frame-skipping cost are far larger. Read the video numbers here as a
floor, not a typical case.
"""
import argparse
import io
import os
import statistics
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "backend"))

BENCH_DIR = os.path.join(ROOT, "eval_data", "bench")
SAMPLE_FACES = os.path.join(ROOT, "training", "tpdn_test")


# ------------------------------------------------------------- measurement

class Timers:
    """Wall time per stage, accumulated across a run."""

    def __init__(self):
        self.total = {}
        self.calls = {}

    def add(self, stage, seconds):
        self.total[stage] = self.total.get(stage, 0.0) + seconds
        self.calls[stage] = self.calls.get(stage, 0) + 1

    def reset(self):
        self.total.clear()
        self.calls.clear()


TIMERS = Timers()


def timed(stage, fn):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            TIMERS.add(stage, time.perf_counter() - start)
    return wrapper


INSIDE_EXPLAIN = threading.local()


def instrument(engine):
    """Wrap the engine's own methods. Idempotent.

    `explain` contains forwards — 36 of them — so timing both naively
    double-counts and the shares sum past 100%. Forwards made while inside
    explain are billed to `explain`, and the stages stay disjoint."""
    if getattr(engine, "_benchmarked", False):
        return

    raw_forward = engine._forward

    def forward(*args, **kwargs):
        start = time.perf_counter()
        try:
            return raw_forward(*args, **kwargs)
        finally:
            elapsed = time.perf_counter() - start
            TIMERS.add("explain" if getattr(INSIDE_EXPLAIN, "on", False)
                       else "forward", elapsed)

    raw_explain = engine.explain

    def explain(*args, **kwargs):
        INSIDE_EXPLAIN.on = True
        start = time.perf_counter()
        try:
            return raw_explain(*args, **kwargs)
        finally:
            INSIDE_EXPLAIN.on = False
            # Only the part of explain that is not forwards; the forwards
            # billed themselves above.
            TIMERS.add("explain", 0.0)
            TIMERS.add("explain-overhead", time.perf_counter() - start)

    engine._detect_face = timed("decode+detect", engine._detect_face)
    engine._normalize_compression = timed("normalise", engine._normalize_compression)
    engine._to_input = timed("prepare", engine._to_input)
    engine._forward = forward
    engine.explain = explain
    engine._benchmarked = True


class RamWatch:
    """Peak RSS during a run, sampled from a thread.

    A before/after delta misses the peak: a sixty-frame video allocates and
    frees as it goes, and the maximum sits somewhere in the middle."""

    def __init__(self, interval=0.02):
        import psutil
        self.proc = psutil.Process()
        self.interval = interval
        self.peak = 0
        self._stop = threading.Event()
        self._thread = None

    def __enter__(self):
        self.baseline = self.proc.memory_info().rss
        self.peak = self.baseline
        self.cpu_before = self.proc.cpu_times()
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def _sample(self):
        while not self._stop.wait(self.interval):
            try:
                self.peak = max(self.peak, self.proc.memory_info().rss)
            except Exception:
                return

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=1)
        after = self.proc.cpu_times()
        self.cpu_seconds = ((after.user - self.cpu_before.user) +
                            (after.system - self.cpu_before.system))


# ---------------------------------------------------------------- fixtures

def ensure(path, build):
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        build(path)
    return path


def faces():
    import glob
    found = sorted(glob.glob(os.path.join(SAMPLE_FACES, "*.jpeg")))
    if not found:
        sys.exit("no sample faces in training/tpdn_test")
    return found


def make_image(side):
    def build(path):
        from PIL import Image
        with Image.open(faces()[0]) as im:
            im.convert("RGB").resize((side, side), Image.LANCZOS).save(
                path, "JPEG", quality=92)
    return build


def make_video(seconds, fps=10, side=480):
    """One encoded clip of `seconds` length. Sampling at 1 fps means the
    engine will look at `seconds` frames, up to its own cap."""
    def build(path):
        import cv2
        imgs = [cv2.resize(cv2.imread(p), (side, side)) for p in faces()[:3]]
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"),
                                 float(fps), (side, side))
        for i in range(seconds * fps):
            writer.write(imgs[i % len(imgs)])
        writer.release()
    return build


def cases(only=None):
    all_cases = [
        ("image-224", "image", ensure(os.path.join(BENCH_DIR, "img_224.jpg"),
                                      make_image(224))),
        ("image-1024", "image", ensure(os.path.join(BENCH_DIR, "img_1024.jpg"),
                                       make_image(1024))),
        ("video-10s", "video", ensure(os.path.join(BENCH_DIR, "clip_10s.mp4"),
                                      make_video(10))),
        ("video-30s", "video", ensure(os.path.join(BENCH_DIR, "clip_30s.mp4"),
                                      make_video(30))),
        ("video-60s", "video", ensure(os.path.join(BENCH_DIR, "clip_60s.mp4"),
                                      make_video(60))),
    ]
    return [c for c in all_cases if not only or c[0] in only]


# --------------------------------------------------------------- the runs

def measure_staging(path, kind):
    """Server-side upload cost: validation plus the write to disk. Network
    time is not included and would dominate on a real connection."""
    import app as app_module
    app_module.app.config.update(TESTING=True)
    app_module.limiter._hits.clear()

    with open(path, "rb") as f:
        blob = f.read()
    with app_module.app.test_client() as client:
        start = time.perf_counter()
        response = client.post("/api/upload", data={
            "file": (io.BytesIO(blob), os.path.basename(path))},
            content_type="multipart/form-data")
        elapsed = time.perf_counter() - start

    body = response.get_json() or {}
    staged = app_module.staged_upload_path(body.get("uploadId", ""))
    if staged:
        os.remove(staged)
    return elapsed, len(blob)


def run_case(name, kind, path, repeats):
    import inference

    engine = inference._get_engine()
    instrument(engine)

    staging, size_bytes = measure_staging(path, kind)

    # One untimed pass so lazy initialisation is not billed to the first run
    inference.analyze_file(path, kind)

    totals, breakdowns, frames = [], [], None
    with RamWatch() as ram:
        for _ in range(repeats):
            TIMERS.reset()
            start = time.perf_counter()
            result = inference.analyze_file(path, kind)
            totals.append(time.perf_counter() - start)
            breakdowns.append(dict(TIMERS.total))
            frames = result.get("framesAnalyzed")

    for b in breakdowns:
        if "explain-overhead" in b:
            b["explain"] = b.pop("explain-overhead")   # forwards already inside

    stages = sorted({k for b in breakdowns for k in b})
    median = {s: statistics.median([b.get(s, 0.0) for b in breakdowns]) for s in stages}

    return {
        "case": name, "kind": kind, "frames": frames,
        "bytes": size_bytes,
        "staging": staging,
        "total": statistics.median(totals),
        "stages": median,
        "peak_rss": ram.peak,
        "baseline_rss": ram.baseline,
        "cpu_seconds": ram.cpu_seconds / repeats,
    }


# ------------------------------------------------------------------ report

def report(rows, repeats):
    mb = lambda b: b / (1024 * 1024)

    print("\n" + "=" * 78)
    print(f"LATENCY   median of {repeats} runs, seconds")
    print("=" * 78)
    stages = ["decode+detect", "normalise", "prepare", "forward", "explain"]
    print(f"  {'case':<12}{'staging':>9}" +
          "".join(f"{s:>15}" for s in stages) + f"{'other':>9}{'TOTAL':>9}")
    print("  " + "-" * 74)
    for r in rows:
        accounted = sum(r["stages"].get(s, 0.0) for s in stages)
        other = max(0.0, r["total"] - accounted)
        line = f"  {r['case']:<12}{r['staging']:>9.3f}"
        line += "".join(f"{r['stages'].get(s, 0.0):>15.3f}" for s in stages)
        print(line + f"{other:>9.3f}{r['total']:>9.3f}")

    print("\n" + "=" * 78)
    print("SHARE OF TOTAL")
    print("=" * 78)
    print(f"  {'case':<12}" + "".join(f"{s:>15}" for s in stages) + f"{'other':>9}")
    print("  " + "-" * 74)
    for r in rows:
        accounted = sum(r["stages"].get(s, 0.0) for s in stages)
        other = max(0.0, r["total"] - accounted)
        pct = lambda v: f"{v / r['total'] * 100:>14.1f}%" if r["total"] else "     -"
        print(f"  {r['case']:<12}" +
              "".join(pct(r["stages"].get(s, 0.0)) for s in stages) +
              f"{other / r['total'] * 100:>8.1f}%")

    print("\n" + "=" * 78)
    print("RESOURCES")
    print("=" * 78)
    print(f"  {'case':<12}{'frames':>8}{'file MB':>10}{'peak RSS MB':>14}"
          f"{'over baseline':>15}{'CPU s/run':>12}{'CPU cores':>11}")
    print("  " + "-" * 74)
    for r in rows:
        cores = r["cpu_seconds"] / r["total"] if r["total"] else 0
        print(f"  {r['case']:<12}{str(r['frames'] or '-'):>8}{mb(r['bytes']):>10.2f}"
              f"{mb(r['peak_rss']):>14.1f}{mb(r['peak_rss'] - r['baseline_rss']):>15.1f}"
              f"{r['cpu_seconds']:>12.2f}{cores:>11.2f}")

    videos = [r for r in rows if r["kind"] == "video" and r["frames"]]
    if len(videos) > 1:
        print("\n" + "=" * 78)
        print("PER FRAME  — does cost scale with length, or is there a fixed price?")
        print("=" * 78)
        for r in videos:
            per = r["total"] / r["frames"]
            print(f"  {r['case']:<12}{r['frames']:>4} frames"
                  f"{per * 1000:>10.1f} ms/frame"
                  f"{r['total']:>10.2f} s total")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--only", nargs="*", help="case names to run")
    args = ap.parse_args()

    import inference
    if not inference.engine_available():
        sys.exit("no model available — nothing to benchmark")

    info = inference.engine_info()
    print(f"DeepShield benchmark — {info.get('model_name')} {info.get('version')} "
          f"({info.get('runtime')}), {os.cpu_count()} cores")

    rows = []
    for name, kind, path in cases(args.only):
        print(f"  running {name} …", end="", flush=True)
        rows.append(run_case(name, kind, path, args.repeats))
        print(f" {rows[-1]['total']:.2f}s")

    report(rows, args.repeats)
    return 0


if __name__ == "__main__":
    sys.exit(main())
