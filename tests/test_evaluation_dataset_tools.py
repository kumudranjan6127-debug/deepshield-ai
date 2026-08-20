"""Tests for the provenance-first evaluation dataset tools."""
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from analyze_failures import analyze
from check_dataset import format_report, inspect
from dataset_common import METADATA_FIELDS, metadata_by_path, read_csv
from dataset_manifest import build_manifest
from summarize_benchmark import summarize


def _write_metadata(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=METADATA_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _image(path, colour=(20, 40, 60), fmt="PNG"):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (160, 120), colour).save(path, fmt)


def _metadata(relative_path, label="real", **extra):
    row = {"relative_path": relative_path, "label": label, "modality": "image",
           "source_dataset": "real_phone" if label == "real" else "dfdc",
           "source_id": "source-1", "identity_id": "person-1",
           "identity_group": "person-1",
           "manipulation_type": "none" if label == "real" else "face_swap",
           "generator_family": "real" if label == "real" else "face_swap",
           "compression_slice": "native", "robustness_slice": "clean",
           "split": "validation", "generator_disjoint": "no",
           "provenance": "authoritative local metadata", "usage_note": "evaluation only"}
    row.update(extra)
    return row


def test_manifest_uses_explicit_metadata_not_directory_labels(tmp_path):
    dataset = tmp_path / "dataset"
    _image(dataset / "real" / "sample.png")
    _write_metadata(dataset / "metadata.csv", [_metadata("real/sample.png", label="fake")])

    rows, errors = build_manifest(dataset, dataset / "metadata.csv", tmp_path / "dataset_manifest.csv")

    assert not errors
    assert rows[0]["label"] == "fake"
    assert rows[0]["source_dataset"] == "dfdc"
    assert rows[0]["sha256"]
    assert rows[0]["width"] == 160 and rows[0]["height"] == 120


def test_checker_reports_duplicates_invalid_missing_metadata_and_identity_overlap(tmp_path):
    dataset = tmp_path / "dataset"
    _image(dataset / "real" / "original.png")
    (dataset / "fake").mkdir()
    shutil.copyfile(dataset / "real" / "original.png", dataset / "fake" / "exact.png")
    _image(dataset / "fake" / "visual.jpg", fmt="JPEG")
    (dataset / "fake" / "broken.jpg").write_bytes(b"not an image")
    _image(dataset / "real" / "unrecorded.png", colour=(1, 2, 3))
    metadata = [
        _metadata("real/original.png", identity_id="overlap-id"),
        _metadata("fake/exact.png", label="fake"),
        _metadata("fake/visual.jpg", label="fake"),
        _metadata("fake/broken.jpg", label="fake", provenance=""),
    ]
    _write_metadata(dataset / "metadata.csv", metadata)
    manifest_path = tmp_path / "dataset_manifest.csv"
    build_manifest(dataset, dataset / "metadata.csv", manifest_path)
    training = tmp_path / "training.csv"
    training.write_text("identity_id\noverlap-id\n", encoding="utf-8")

    report = inspect(dataset, read_csv(manifest_path), training)

    assert report["duplicate_count"] >= 2
    assert report["duplicate_sha256_groups"]
    assert report["duplicate_image_groups"]
    assert report["invalid_file_count"] == 1
    assert report["missing_metadata_count"] >= 2
    assert report["missing_provenance_count"] >= 1
    assert report["identity_overlap_status"] == "identity overlap detected"
    assert report["overlapping_identity_ids"] == ["overlap-id"]
    assert "DeepShield evaluation dataset quality report" in format_report(report)


def test_report_marks_identity_disjointness_unverified_without_training_metadata(tmp_path):
    rows = [_metadata("real/a.jpg")]

    report = inspect(tmp_path, rows)

    assert report["identity_overlap_status"] == "identity-disjointness cannot be verified"


def test_subgroup_report_keeps_face_swap_and_phone_metadata_separate():
    manifest = [
        _metadata("real/phone.jpg"),
        _metadata("fake/swap.jpg", label="fake", manipulation_type="face_swap"),
        _metadata("fake/other.jpg", label="fake", manipulation_type="face_reenactment"),
    ]
    predictions = [
        {"file": "real/phone.jpg", "prediction": "real", "confidence": "95", "face_found": "True", "inconclusive": "False", "latency_ms": "10", "error": ""},
        {"file": "fake/swap.jpg", "prediction": "real", "confidence": "92", "face_found": "True", "inconclusive": "False", "latency_ms": "20", "error": ""},
        {"file": "fake/other.jpg", "prediction": "deepfake", "confidence": "96", "face_found": "True", "inconclusive": "False", "latency_ms": "30", "error": ""},
    ]

    report = summarize(predictions, manifest)

    assert report["groups"]["all_samples"]["confusion_matrix"] == {"tp": 1, "tn": 1, "fp": 0, "fn": 1}
    assert report["groups"]["face_swap_samples"]["false_negative_rate"] == 1.0
    assert report["groups"]["real_phone_photo_samples"]["samples"] == 1


def test_unsafe_metadata_paths_are_rejected_instead_of_rewritten(tmp_path):
    metadata = tmp_path / "metadata.csv"
    _write_metadata(metadata, [
        _metadata("../real/sample.png"),
        _metadata("/real/sample.png"),
        _metadata("C:\\real\\sample.png"),
    ])

    index, errors = metadata_by_path(metadata)

    assert index == {}
    assert len(errors) == 3
    assert all("unsafe relative_path" in error for error in errors)


def test_manifest_rejects_symlinks_that_escape_dataset(tmp_path):
    dataset = tmp_path / "dataset"
    outside = tmp_path / "outside.png"
    _image(outside)
    dataset.mkdir()
    link = dataset / "escape.png"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("symlinks are unavailable on this platform")
    _write_metadata(dataset / "metadata.csv", [_metadata("escape.png")])

    rows, errors = build_manifest(
        dataset, dataset / "metadata.csv", tmp_path / "manifest.csv"
    )

    assert rows == []
    assert any("resolves outside dataset" in error for error in errors)


def test_manifest_cli_fails_and_writes_machine_readable_errors(tmp_path):
    dataset = tmp_path / "dataset"
    _image(dataset / "real" / "sample.png")
    _write_metadata(dataset / "metadata.csv", [_metadata("missing.png")])
    output = tmp_path / "manifest.csv"

    completed = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "dataset_manifest.py"),
         "--dataset", str(dataset), "--out", str(output)],
        text=True, capture_output=True, check=False,
    )

    assert completed.returncode == 2
    assert output.is_file()
    report = json.loads(
        output.with_suffix(output.suffix + ".errors.json").read_text(encoding="utf-8")
    )
    assert report["ok"] is False
    assert report["errors"]


def test_inference_errors_are_not_misreported_as_no_face():
    report = analyze(
        [{
            "file": "real/sample.png",
            "prediction": "",
            "confidence": "",
            "face_found": "",
            "inconclusive": "False",
            "error": "RuntimeError: inference failed",
        }],
        [_metadata("real/sample.png")],
    )

    assert report["counts"]["no_face"] == 0
