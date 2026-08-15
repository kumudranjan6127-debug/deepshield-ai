#!/usr/bin/env python3
"""Benchmark the currently installed DeepShield image model.

Dataset layout:
    dataset/
      real/        authentic images
      fake/        manipulated/deepfake images

Run from the repository root:
    python tools/benchmark_model.py --dataset dataset

The script never uses filenames as model input and records each prediction,
confidence, latency, and failure. It is intentionally independent of training
so V3/V4 models can be compared with the same evaluation protocol.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path

# Allow execution as `python tools/benchmark_model.py` from repo root.
ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from inference import analyze_file, engine_available, engine_info  # noqa: E402

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def collect(root: Path, label: str):
    folder = root / label
    if not folder.exists():
        return []
    return [(p, label) for p in sorted(folder.rglob("*")) if p.is_file() and p.suffix.lower() in EXTS]


def metrics(rows):
    successful = [r for r in rows if r["error"] == ""]
    valid = [r for r in successful if not r["inconclusive"]]
    tp = sum(r["label"] == "fake" and r["prediction"] == "deepfake" for r in valid)
    tn = sum(r["label"] == "real" and r["prediction"] == "real" for r in valid)
    fp = sum(r["label"] == "real" and r["prediction"] == "deepfake" for r in valid)
    fn = sum(r["label"] == "fake" and r["prediction"] == "real" for r in valid)
    inconclusive = sum(r["inconclusive"] for r in successful)
    n = len(valid)
    accuracy = (tp + tn) / n if n else None
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * precision * recall / (precision + recall) if precision is not None and recall is not None and precision + recall else None
    fpr = fp / (fp + tn) if fp + tn else None
    fnr = fn / (fn + tp) if fn + tp else None
    return {
        "samples": len(rows), "successful": len(successful), "evaluated": n,
        "errors": len(rows) - len(successful),
        "inconclusive_no_face": inconclusive,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy": round(accuracy, 4) if accuracy is not None else None,
        "precision": round(precision, 4) if precision is not None else None,
        "recall": round(recall, 4) if recall is not None else None,
        "f1": round(f1, 4) if f1 is not None else None,
        "false_positive_rate": round(fpr, 4) if fpr is not None else None,
        "false_negative_rate": round(fnr, 4) if fnr is not None else None,
    }


def latency_summary(rows):
    latencies = [r["latency_ms"] for r in rows if isinstance(r["latency_ms"], (int, float))]
    if not latencies:
        return {"mean": None, "median": None, "p95": None}
    ordered = sorted(latencies)
    p95_index = max(0, (len(ordered) * 95 + 99) // 100 - 1)
    return {
        "mean": round(statistics.mean(latencies), 2),
        "median": round(statistics.median(latencies), 2),
        "p95": round(ordered[p95_index], 2),
    }


def human_summary(report):
    """A compact, reviewable companion to the JSON report."""
    m = report["metrics"]
    latency = report["latency_ms"]

    def percent(value):
        return "n/a" if value is None else f"{value * 100:.2f}%"

    return "\n".join([
        "DeepShield model benchmark",
        f"Model: {report['model'].get('model_name', 'unknown')} "
        f"{report['model'].get('version', '')} ({report['model'].get('runtime', '')})".strip(),
        f"Samples: {m['samples']} total; {m['evaluated']} conclusive; "
        f"{m['inconclusive_no_face']} inconclusive/no-face; {m['errors']} errors",
        f"Confusion matrix: TP={m['tp']} TN={m['tn']} FP={m['fp']} FN={m['fn']}",
        f"Accuracy: {percent(m['accuracy'])}",
        f"Precision: {percent(m['precision'])}",
        f"Recall: {percent(m['recall'])}",
        f"F1: {percent(m['f1'])}",
        f"False-positive rate: {percent(m['false_positive_rate'])}",
        f"False-negative rate: {percent(m['false_negative_rate'])}",
        f"Latency (ms): mean={latency['mean'] if latency['mean'] is not None else 'n/a'} "
        f"median={latency['median'] if latency['median'] is not None else 'n/a'} "
        f"p95={latency['p95'] if latency['p95'] is not None else 'n/a'}",
        "",
        "Metrics exclude inconclusive/no-face samples and processing errors.",
        "Confidence is the current model's winning-class confidence, not calibrated probability.",
    ]) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, type=Path)
    ap.add_argument("--out", default="benchmark-results", type=Path)
    args = ap.parse_args()

    if not engine_available():
        raise SystemExit("No live DeepShield checkpoint/ONNX model is installed. Benchmark requires a real model.")

    samples = collect(args.dataset, "real") + collect(args.dataset, "fake")
    if not samples:
        raise SystemExit("No images found. Expected <dataset>/real and <dataset>/fake.")

    rows = []
    for path, label in samples:
        started = time.perf_counter()
        try:
            # This is the public serving entry point, not a parallel scoring
            # implementation. The path string locates the image; only its
            # decoded pixels reach the detector and model.
            result = analyze_file(str(path), "image")
            elapsed_ms = (time.perf_counter() - started) * 1000
            rows.append({
                "file": str(path.relative_to(args.dataset)),
                "label": label,
                "prediction": result.get("prediction", ""),
                "confidence": result.get("confidence", ""),
                "face_found": bool(result.get("faceFound", False)),
                "face_count": result.get("facesFound", ""),
                # Older current-engine responses report no face with
                # `faceFound: false` but no explicit insufficient-evidence
                # flag. Treat both forms identically in the benchmark.
                "inconclusive": bool(result.get(
                    "insufficientEvidence", not result.get("faceFound", False))),
                "latency_ms": round(elapsed_ms, 2),
                "error": "",
            })
        except Exception as exc:
            rows.append({
                "file": str(path.relative_to(args.dataset)), "label": label,
                "prediction": "", "confidence": "", "face_found": "",
                "face_count": "", "inconclusive": False, "latency_ms": "",
                "error": f"{type(exc).__name__}: {exc}",
            })

    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "predictions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    report = {
        "model": engine_info(),
        "protocol": {
            "labels": {"real": "authentic", "fake": "manipulated/deepfake"},
            "filename_leakage": False,
            "inference_api": "inference.analyze_file(path, 'image')",
            "decision_threshold": 0.5,
            "extensions": sorted(EXTS),
        },
        "metrics": metrics(rows),
        "latency_ms": latency_summary(rows),
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    summary = human_summary(report)
    (args.out / "summary.txt").write_text(summary, encoding="utf-8")
    print(summary)
    print(f"Machine-readable report: {args.out / 'report.json'}")


if __name__ == "__main__":
    main()
