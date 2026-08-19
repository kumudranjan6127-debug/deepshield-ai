#!/usr/bin/env python3
"""Join unchanged benchmark output to provenance metadata for subgroup metrics."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from dataset_common import read_csv

FACE_SWAP_TYPES = {"face_swap", "face-swap", "deepfakes", "faceswap", "faceshifter", "celeb-synthesis", "dfdc"}


def _bool(value) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _metrics(rows):
    usable = [row for row in rows if not row.get("error") and not _bool(row.get("inconclusive"))]
    tp = sum(row["label"] == "fake" and row.get("prediction") == "deepfake" for row in usable)
    tn = sum(row["label"] == "real" and row.get("prediction") == "real" for row in usable)
    fp = sum(row["label"] == "real" and row.get("prediction") == "deepfake" for row in usable)
    fn = sum(row["label"] == "fake" and row.get("prediction") == "real" for row in usable)
    n = len(usable)
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    latencies = [float(row["latency_ms"]) for row in rows if str(row.get("latency_ms", "")).strip()]
    ordered = sorted(latencies)
    p95 = ordered[max(0, (len(ordered) * 95 + 99) // 100 - 1)] if ordered else None
    return {
        "samples": len(rows), "evaluated": n,
        "inconclusive_no_face": sum(_bool(row.get("inconclusive")) for row in rows),
        "errors": sum(bool(row.get("error")) for row in rows),
        "confusion_matrix": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "accuracy": round((tp + tn) / n, 4) if n else None,
        "precision": round(precision, 4) if precision is not None else None,
        "recall": round(recall, 4) if recall is not None else None,
        "f1": round(2 * precision * recall / (precision + recall), 4)
              if precision is not None and recall is not None and precision + recall else None,
        "false_positive_rate": round(fp / (fp + tn), 4) if fp + tn else None,
        "false_negative_rate": round(fn / (fn + tp), 4) if fn + tp else None,
        "latency_ms": {"mean": round(statistics.mean(latencies), 2) if latencies else None,
                       "p50": round(statistics.median(latencies), 2) if latencies else None,
                       "p95": round(p95, 2) if p95 is not None else None},
    }


def summarize(predictions: list[dict], manifest: list[dict]):
    metadata = {row["relative_path"]: row for row in manifest}
    joined, unmatched = [], []
    for prediction in predictions:
        row = {**prediction, **metadata.get(prediction.get("file", ""), {})}
        if prediction.get("file", "") not in metadata:
            unmatched.append(prediction.get("file", ""))
        joined.append(row)
    face_swap = [row for row in joined if row.get("label") == "fake" and
                 row.get("manipulation_type", "").strip().lower() in FACE_SWAP_TYPES]
    other_fake = [row for row in joined if row.get("label") == "fake" and row not in face_swap]
    phone = [row for row in joined if row.get("label") == "real" and
             row.get("source_dataset", "").strip().lower() == "real_phone"]
    return {
        "groups": {
            "all_samples": _metrics(joined),
            "real_samples": _metrics([row for row in joined if row.get("label") == "real"]),
            "fake_samples": _metrics([row for row in joined if row.get("label") == "fake"]),
            "face_swap_samples": _metrics(face_swap),
            "other_manipulation_samples": _metrics(other_fake),
            "real_phone_photo_samples": _metrics(phone),
        },
        "unmatched_prediction_paths": unmatched,
        "note": "Subgroups are provenance metadata joins; they do not change the V3 decision threshold.",
    }


def format_summary(report):
    lines = ["DeepShield V3 subgroup benchmark summary"]
    for name, metric in report["groups"].items():
        cm = metric["confusion_matrix"]
        lines += ["", name + ":", f"  samples={metric['samples']} evaluated={metric['evaluated']} "
                  f"inconclusive={metric['inconclusive_no_face']} errors={metric['errors']}",
                  f"  accuracy={metric['accuracy']} precision={metric['precision']} recall={metric['recall']} f1={metric['f1']}",
                  f"  fpr={metric['false_positive_rate']} fnr={metric['false_negative_rate']} "
                  f"confusion=TP:{cm['tp']} TN:{cm['tn']} FP:{cm['fp']} FN:{cm['fn']}",
                  f"  latency_ms={metric['latency_ms']}"]
    if report["unmatched_prediction_paths"]:
        lines += ["", "Predictions with no manifest metadata:"] + [f"  {path}" for path in report["unmatched_prediction_paths"]]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=Path("benchmark-results/predictions.csv"))
    parser.add_argument("--manifest", type=Path, default=Path("dataset_manifest.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("benchmark-results"))
    args = parser.parse_args()
    if not args.predictions.is_file() or not args.manifest.is_file():
        raise SystemExit("Both predictions.csv and dataset_manifest.csv are required.")
    report = summarize(read_csv(args.predictions), read_csv(args.manifest))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "baseline_subgroups.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    text = format_summary(report)
    (args.out_dir / "baseline_subgroups.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
