#!/usr/bin/env python3
"""Build an auditable manifest for a locally acquired evaluation dataset.

The expected ``dataset/real`` and ``dataset/fake`` directories organise files
for the binary benchmark.  They do *not* label the files here.  Ground truth
and provenance must be entered explicitly in ``dataset/metadata.csv``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from dataset_common import (
    IMAGE_EXTENSIONS,
    MANIFEST_FIELDS,
    VIDEO_EXTENSIONS,
    image_details,
    manifest_validation_errors,
    metadata_by_path,
    normalized_path,
    read_csv,
    safe_media_path,
    sha256_of,
    write_csv,
)


def media_files(dataset: Path, excluded: set[Path], validation_errors: list[str]):
    dataset_root = dataset.resolve()
    for path in sorted(dataset.rglob("*")):
        if not path.is_file():
            continue
        resolved = path.resolve()
        if not resolved.is_relative_to(dataset_root):
            validation_errors.append(
                f"file resolves outside dataset: {path.relative_to(dataset)}"
            )
            continue
        if resolved not in excluded:
            yield path


def build_manifest(dataset: Path, metadata_path: Path, output_path: Path):
    metadata, metadata_errors = metadata_by_path(metadata_path)
    excluded = {metadata_path.resolve(), output_path.resolve()}
    rows = []

    for path in media_files(dataset, excluded, metadata_errors):
        relative = normalized_path(path.relative_to(dataset))
        meta = metadata.get(relative, {})
        errors = []
        _, path_error = safe_media_path(dataset, relative)
        if path_error:
            errors.append(path_error)
        if not meta:
            errors.append("missing metadata")
        extension = path.suffix.lower()
        media_type = ("image" if extension in IMAGE_EXTENSIONS else
                      "video" if extension in VIDEO_EXTENSIONS else "unsupported")
        width = height = ""
        if media_type == "image" and not path_error:
            width, height, image_error = image_details(path)
            if image_error:
                errors.append(image_error)
        elif media_type == "video":
            # The manifest can inventory videos even though the current image
            # benchmark cannot score them as independent frames.
            pass
        else:
            errors.append(f"unsupported format: {extension or '(no extension)'}")

        for field in ("label", "source_dataset", "provenance", "modality",
                      "robustness_slice", "split", "generator_disjoint"):
            if not meta.get(field, ""):
                errors.append(f"missing metadata: {field}")
        if not meta.get("identity_group", "") and not meta.get("identity_id", ""):
            errors.append("missing metadata: identity_group")
        label = meta.get("label", "").lower()
        if label and label not in {"real", "fake"}:
            errors.append(f"invalid metadata label: {label}")
        if label == "fake":
            for field in ("manipulation_type", "generator_family"):
                if not meta.get(field, ""):
                    errors.append(f"missing metadata for fake sample: {field}")
        if meta.get("modality", "").lower() and meta["modality"].lower() != media_type:
            errors.append(
                f"metadata modality {meta['modality']!r} does not match file type {media_type!r}")
        status = "ok" if not errors else "; ".join(errors)
        rows.append({
            "relative_path": relative,
            "label": label,
            "media_type": media_type,
            "modality": meta.get("modality", "").lower(),
            "source_dataset": meta.get("source_dataset", ""),
            "source_id": meta.get("source_id", ""),
            "identity_id": meta.get("identity_id", ""),
            "identity_group": meta.get("identity_group", ""),
            "manipulation_type": meta.get("manipulation_type", ""),
            "generator_family": meta.get("generator_family", ""),
            "compression_slice": meta.get("compression_slice", ""),
            "robustness_slice": meta.get("robustness_slice", ""),
            "split": meta.get("split", "").lower(),
            "generator_disjoint": meta.get("generator_disjoint", "").lower(),
            "provenance": meta.get("provenance", ""),
            "usage_note": meta.get("usage_note", ""),
            "sha256": sha256_of(path) if not path_error else "",
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
    for relative in sorted(metadata):
        _, path_error = safe_media_path(dataset, relative)
        if path_error:
            metadata_errors.append(f"metadata path {relative!r}: {path_error}")

    # Write the full ledger even when it is invalid; dropping bad rows makes
    # a later review unable to tell what was excluded. The command still
    # returns non-zero below, so this cannot accidentally become a benchmark.
    # Hashes were computed while building each row. Avoid reading large videos
    # a second time here; --validate and evaluation re-check stored hashes.
    metadata_errors.extend(manifest_validation_errors(
        rows, dataset, verify_hashes=False))
    write_csv(output_path, MANIFEST_FIELDS, rows)
    return rows, metadata_errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("dataset"))
    parser.add_argument("--metadata", type=Path, default=None,
                        help="CSV with explicit labels and provenance (default: DATASET/metadata.csv)")
    parser.add_argument("--out", type=Path, default=Path("dataset_manifest.csv"))
    parser.add_argument("--validate", action="store_true",
                        help="validate an existing --out manifest without rebuilding it")
    args = parser.parse_args()

    if not args.dataset.is_dir():
        parser.error(f"Dataset directory not found: {args.dataset}")
    if args.validate:
        if not args.out.is_file():
            parser.error(f"Manifest not found: {args.out}")
        rows = read_csv(args.out)
        errors = manifest_validation_errors(rows, args.dataset)
    else:
        metadata = args.metadata or args.dataset / "metadata.csv"
        rows, errors = build_manifest(args.dataset, metadata, args.out)
    print(f"Manifest: {args.out} ({len(rows)} files)")
    if errors:
        print("Manifest validation issues:")
        for error in errors:
            print(f"  - {error}")
    if errors:
        error_report = args.out.with_suffix(args.out.suffix + ".errors.json")
        error_report.write_text(
            json.dumps({"ok": False, "errors": errors}, indent=2),
            encoding="utf-8",
        )
        return 2
    print("Manifest validation: passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
