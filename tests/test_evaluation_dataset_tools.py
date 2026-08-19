"""Tests for the provenance-first evaluation dataset tools."""
import csv
import shutil
import sys
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from check_dataset import format_report, inspect
from dataset_common import METADATA_FIELDS, read_csv
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
    row = {"relative_path": relative_path, "label": label,
           "source_dataset": "real_phone" if label == "real" else "dfdc",
           "source_id": "source-1", "identity_id": "person-1",
           "manipulation_type": "none" if label == "real" else "face_swap",
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
    assert report["missing_provenance_count"] == 1
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
