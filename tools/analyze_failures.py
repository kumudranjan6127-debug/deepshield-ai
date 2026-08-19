#!/usr/bin/env python3
"""Produce a metadata-only failure ledger from benchmark predictions."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from dataset_common import read_csv
from summarize_benchmark import _bool

FIELDS = ["category", "file_path", "true_label", "predicted_label", "confidence",
          "face_count", "source_dataset", "manipulation_type", "latency_ms"]


def analyze(predictions, manifest, high_confidence=90, low_confidence=70):
    metadata = {row["relative_path"]: row for row in manifest}
    records = []
    for prediction in predictions:
        meta = metadata.get(prediction.get("file", ""), {})
        label, predicted = meta.get("label", ""), prediction.get("prediction", "")
        try:
            confidence = int(prediction.get("confidence", ""))
        except (TypeError, ValueError):
            confidence = None
        categories = []
        if _bool(prediction.get("inconclusive")) or not _bool(prediction.get("face_found")):
            categories.append("no_face")
        if label == "real" and predicted == "deepfake":
            categories.append("false_positive")
        if label == "fake" and predicted == "real":
            categories.append("false_negative")
        if categories and ("false_positive" in categories or "false_negative" in categories) and confidence is not None and confidence >= high_confidence:
            categories.append("high_confidence_wrong")
        if confidence is not None and confidence <= low_confidence:
            categories.append("low_confidence")
        for category in categories:
            records.append({"category": category, "file_path": prediction.get("file", ""),
                            "true_label": label, "predicted_label": predicted,
                            "confidence": confidence if confidence is not None else "",
                            "face_count": prediction.get("face_count", ""),
                            "source_dataset": meta.get("source_dataset", ""),
                            "manipulation_type": meta.get("manipulation_type", ""),
                            "latency_ms": prediction.get("latency_ms", "")})
    counts = {name: sum(row["category"] == name for row in records)
              for name in ("false_positive", "false_negative", "high_confidence_wrong", "no_face", "low_confidence")}
    return {"counts": counts, "records": records,
            "reporting_thresholds": {"high_confidence": high_confidence, "low_confidence": low_confidence}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=Path("benchmark-results/predictions.csv"))
    parser.add_argument("--manifest", type=Path, default=Path("dataset_manifest.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("benchmark-results"))
    parser.add_argument("--high-confidence", type=int, default=90)
    parser.add_argument("--low-confidence", type=int, default=70)
    args = parser.parse_args()
    if not args.predictions.is_file() or not args.manifest.is_file():
        raise SystemExit("Both predictions.csv and dataset_manifest.csv are required.")
    report = analyze(read_csv(args.predictions), read_csv(args.manifest), args.high_confidence, args.low_confidence)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "failure_analysis.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader(); writer.writerows(report["records"])
    (args.out_dir / "failure_analysis.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    text = "DeepShield failure analysis\n" + "\n".join(f"{key}: {value}" for key, value in report["counts"].items()) + "\n"
    (args.out_dir / "failure_analysis.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
