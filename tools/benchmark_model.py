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

from inference import engine_available, _get_engine  # noqa: E402

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def collect(root: Path, label: str):
    folder = root / label
    if not folder.exists():
        return []
    return [(p, label) for p in sorted(folder.rglob("*")) if p.is_file() and p.suffix.lower() in EXTS]


def metrics(rows):
    valid = [r for r in rows if r["error"] == ""]
    tp = sum(r["label"] == "fake" and r["prediction"] == "deepfake" for r in valid)
    tn = sum(r["label"] == "real" and r["prediction"] == "real" for r in valid)
    fp = sum(r["label"] == "real" and r["prediction"] == "deepfake" for r in valid)
    fn = sum(r["label"] == "fake" and r["prediction"] == "real" for r in valid)
    inconclusive = sum(r["insufficient"] for r in valid)
    n = len(valid)
    accuracy = (tp + tn) / n if n else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    fnr = fn / (fn + tp) if fn + tp else 0.0
    return {
        "samples": len(rows), "evaluated": n, "errors": len(rows) - n,
        "inconclusive_no_face": inconclusive,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy": round(accuracy, 4), "precision": round(precision, 4),
        "recall": round(recall, 4), "f1": round(f1, 4),
        "false_positive_rate": round(fpr, 4), "false_negative_rate": round(fnr, 4),
    }


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

    engine = _get_engine()
    rows = []
    for path, label in samples:
        started = time.perf_counter()
        try:
            result = engine.predict_image(str(path))
            elapsed_ms = (time.perf_counter() - started) * 1000
            rows.append({
                "file": str(path.relative_to(args.dataset)),
                "label": label,
                "prediction": result.get("prediction", ""),
                "confidence": result.get("confidence", ""),
                "insufficient": bool(result.get("insufficientEvidence", False)),
                "faces": result.get("facesFound", ""),
                "latency_ms": round(elapsed_ms, 2),
                "error": "",
            })
        except Exception as exc:
            rows.append({
                "file": str(path.relative_to(args.dataset)), "label": label,
                "prediction": "", "confidence": "", "insufficient": False,
                "faces": "", "latency_ms": "", "error": f"{type(exc).__name__}: {exc}",
            })

    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "predictions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    latencies = [r["latency_ms"] for r in rows if isinstance(r["latency_ms"], (int, float))]
    report = {
        "model": engine.info,
        "protocol": {
            "labels": {"real": "authentic", "fake": "manipulated/deepfake"},
            "filename_leakage": False,
            "extensions": sorted(EXTS),
        },
        "metrics": metrics(rows),
        "latency_ms": {
            "mean": round(statistics.mean(latencies), 2) if latencies else None,
            "median": round(statistics.median(latencies), 2) if latencies else None,
            "p95": round(sorted(latencies)[max(0, int(len(latencies) * .95) - 1)], 2) if latencies else None,
        },
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
