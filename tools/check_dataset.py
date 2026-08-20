#!/usr/bin/env python3
"""Validate an evaluation manifest and write a provenance quality report."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from dataset_common import (IMAGE_EXTENSIONS, MANIFEST_FIELDS, dhash_of,
                            distribution, duplicate_groups, image_details,
                            read_csv, resolution_distribution, sha256_of)


def _safe_path(dataset: Path, relative: str) -> Path | None:
    candidate = (dataset / relative).resolve()
    try:
        candidate.relative_to(dataset.resolve())
    except ValueError:
        return None
    return candidate


def _training_identities(path: Path | None):
    if not path or not path.exists():
        return set(), False
    rows = read_csv(path)
    fields = set(rows[0]) if rows else set()
    key = "identity_id" if "identity_id" in fields else "subject_id" if "subject_id" in fields else None
    if not key:
        return set(), False
    ids = {str(row.get(key, "") or "").strip() for row in rows}
    return ids - {""}, bool(ids - {""})


def _identity_status(rows, training_manifest):
    evaluation_ids = {str(row.get("identity_id", "") or "").strip() for row in rows}
    evaluation_ids.discard("")
    training_ids, train_available = _training_identities(training_manifest)
    if not evaluation_ids or not train_available:
        return "identity-disjointness cannot be verified", []
    overlap = sorted(evaluation_ids & training_ids)
    if overlap:
        return "identity overlap detected", overlap
    return "no identity overlap detected against supplied training manifest", []


def inspect(dataset: Path, rows: list[dict], training_manifest: Path | None = None):
    """Return a pure JSON-compatible validation report without modifying media."""
    problems = []
    hashes, image_hashes = {}, {}
    invalid_files, unsupported_files, missing_metadata = [], [], []

    for row in rows:
        relative = str(row.get("relative_path", "") or "")
        row_issues = []
        for field in ("label", "source_dataset", "provenance"):
            if not str(row.get(field, "") or "").strip():
                row_issues.append(f"missing metadata: {field}")
        if row.get("label", "").lower() not in {"real", "fake"}:
            row_issues.append("missing or invalid label")

        path = _safe_path(dataset, relative)
        if not path or not path.is_file():
            row_issues.append("file missing or outside dataset")
            invalid_files.append(relative)
        elif path.suffix.lower() not in IMAGE_EXTENSIONS:
            row_issues.append("unsupported format")
            unsupported_files.append(relative)
        else:
            width, height, image_error = image_details(path)
            if image_error:
                row_issues.append(image_error)
                invalid_files.append(relative)
            else:
                actual_hash = sha256_of(path)
                hashes[relative] = actual_hash
                if row.get("sha256") and row["sha256"] != actual_hash:
                    row_issues.append("manifest sha256 does not match file")
                try:
                    image_hashes[relative] = dhash_of(path)
                except Exception as exc:
                    row_issues.append(f"could not hash image pixels: {type(exc).__name__}")

        if any(issue.startswith("missing") for issue in row_issues):
            missing_metadata.append(relative)
        if row_issues:
            problems.append({"relative_path": relative, "issues": row_issues})

    exact = duplicate_groups(hashes)
    visual = [group for group in duplicate_groups(image_hashes)
              if len({hashes.get(item) for item in group}) > 1]
    duplicate_paths = {item for group in exact + visual for item in group}
    identity_status, overlapping_ids = _identity_status(rows, training_manifest)
    labels = distribution(rows, "label")

    return {
        "total_samples": len(rows),
        "real_count": labels.get("real", 0),
        "fake_count": labels.get("fake", 0),
        "manipulation_type_distribution": distribution(rows, "manipulation_type"),
        "source_distribution": distribution(rows, "source_dataset"),
        "resolution_distribution": resolution_distribution(rows),
        "duplicate_count": len(duplicate_paths),
        "duplicate_sha256_groups": exact,
        "duplicate_image_groups": visual,
        "invalid_file_count": len(set(invalid_files)),
        "invalid_files": sorted(set(invalid_files)),
        "unsupported_format_count": len(set(unsupported_files)),
        "unsupported_files": sorted(set(unsupported_files)),
        "missing_provenance_count": sum(not str(row.get("provenance", "") or "").strip() for row in rows),
        "missing_metadata_count": len(set(missing_metadata)),
        "missing_metadata_files": sorted(set(missing_metadata)),
        "identity_overlap_status": identity_status,
        "overlapping_identity_ids": overlapping_ids,
        "problems": problems,
    }


def format_report(report: dict) -> str:
    def table(values):
        return "\n".join(f"  {key}: {value}" for key, value in values.items()) or "  (none)"
    return "\n".join([
        "DeepShield evaluation dataset quality report",
        f"Total samples: {report['total_samples']}",
        f"Real: {report['real_count']}", f"Fake: {report['fake_count']}",
        f"Duplicate files/images: {report['duplicate_count']}",
        f"Invalid files: {report['invalid_file_count']}",
        f"Unsupported formats: {report['unsupported_format_count']}",
        f"Missing provenance: {report['missing_provenance_count']}",
        f"Missing required metadata: {report['missing_metadata_count']}",
        f"Identity overlap: {report['identity_overlap_status']}",
        "", "Sources:", table(report["source_distribution"]),
        "", "Manipulation types:", table(report["manipulation_type_distribution"]),
        "", "Resolution buckets:", table(report["resolution_distribution"]), "",
    ])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("dataset"))
    parser.add_argument("--manifest", type=Path, default=Path("dataset_manifest.csv"))
    parser.add_argument("--training-manifest", type=Path,
                        help="Optional training manifest with identity_id or subject_id")
    parser.add_argument("--out-dir", type=Path, default=Path("benchmark-results"))
    args = parser.parse_args()

    if not args.manifest.is_file():
        raise SystemExit(f"Manifest not found: {args.manifest}. Run tools/dataset_manifest.py first.")
    rows = read_csv(args.manifest)
    missing = set(MANIFEST_FIELDS) - set(rows[0] if rows else [])
    if missing:
        raise SystemExit(f"Manifest is missing columns: {', '.join(sorted(missing))}")
    report = inspect(args.dataset, rows, args.training_manifest)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "dataset_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (args.out_dir / "dataset_report.txt").write_text(format_report(report), encoding="utf-8")
    print(format_report(report))


if __name__ == "__main__":
    main()
