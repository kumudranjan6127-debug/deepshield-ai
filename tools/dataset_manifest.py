#!/usr/bin/env python3
"""Build an auditable manifest for a locally acquired evaluation dataset.

The expected ``dataset/real`` and ``dataset/fake`` directories organise files
for the binary benchmark.  They do *not* label the files here.  Ground truth
and provenance must be entered explicitly in ``dataset/metadata.csv``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from dataset_common import (IMAGE_EXTENSIONS, MANIFEST_FIELDS, image_details,
                            metadata_by_path, normalized_path, sha256_of,
                            write_csv)


def media_files(dataset: Path, excluded: set[Path]):
    for path in sorted(dataset.rglob("*")):
        if path.is_file() and path.resolve() not in excluded:
            yield path


def build_manifest(dataset: Path, metadata_path: Path, output_path: Path):
    metadata, metadata_errors = metadata_by_path(metadata_path)
    excluded = {metadata_path.resolve(), output_path.resolve()}
    rows = []

    for path in media_files(dataset, excluded):
        relative = normalized_path(path.relative_to(dataset))
        meta = metadata.get(relative, {})
        errors = []
        if not meta:
            errors.append("missing metadata")
        extension = path.suffix.lower()
        media_type = "image" if extension in IMAGE_EXTENSIONS else "unsupported"
        width = height = ""
        if media_type == "image":
            width, height, image_error = image_details(path)
            if image_error:
                errors.append(image_error)
        else:
            errors.append(f"unsupported format: {extension or '(no extension)'}")

        for field in ("label", "source_dataset", "provenance"):
            if not meta.get(field, ""):
                errors.append(f"missing metadata: {field}")
        label = meta.get("label", "").lower()
        if label and label not in {"real", "fake"}:
            errors.append(f"invalid metadata label: {label}")

        status = "ok" if not errors else "; ".join(errors)
        rows.append({
            "relative_path": relative,
            "label": label,
            "media_type": media_type,
            "source_dataset": meta.get("source_dataset", ""),
            "source_id": meta.get("source_id", ""),
            "identity_id": meta.get("identity_id", ""),
            "manipulation_type": meta.get("manipulation_type", ""),
            "provenance": meta.get("provenance", ""),
            "usage_note": meta.get("usage_note", ""),
            "sha256": sha256_of(path),
            "width": width,
            "height": height,
            "status": "ok" if status == "ok" else "needs_review",
            "validation_errors": " | ".join(errors),
        })

    # Metadata-only entries are just as important as unknown files: preserve
    # them as an error instead of silently pretending every declaration exists.
    discovered = {row["relative_path"] for row in rows}
    metadata_errors.extend(
        f"metadata references no local file: {relative}"
        for relative in sorted(set(metadata) - discovered)
    )
    write_csv(output_path, MANIFEST_FIELDS, rows)
    return rows, metadata_errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("dataset"))
    parser.add_argument("--metadata", type=Path, default=None,
                        help="CSV with explicit labels and provenance (default: DATASET/metadata.csv)")
    parser.add_argument("--out", type=Path, default=Path("dataset_manifest.csv"))
    args = parser.parse_args()

    metadata = args.metadata or args.dataset / "metadata.csv"
    if not args.dataset.is_dir():
        raise SystemExit(f"Dataset directory not found: {args.dataset}")
    rows, errors = build_manifest(args.dataset, metadata, args.out)
    print(f"Manifest: {args.out} ({len(rows)} files)")
    if errors:
        print("Metadata issues:")
        for error in errors:
            print(f"  - {error}")


if __name__ == "__main__":
    main()
