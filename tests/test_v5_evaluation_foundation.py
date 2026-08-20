"""Regression coverage for the V5 evaluation and calibration gate."""
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "scripts"))

from dataset_common import (
    MANIFEST_FIELDS,
    manifest_validation_errors,
    relative_path_error,
    safe_media_path,
    write_csv,
)
from dataset_manifest import main as manifest_main
from ds_metrics import (
    apply_temperature,
    bootstrap_confidence_intervals,
    brier,
    confusion,
    decide_with_abstention,
    ece,
    evaluate,
    fit_temperature,
    pr_auc,
    reliability,
    roc_auc,
    select_abstention_thresholds,
)
from evaluate import (
    ARTIFACT_SCHEMA_VERSION,
    CALIBRATION_ARTIFACT,
    CLASS_CONVENTION,
    THRESHOLD_ARTIFACT,
    artifact_context,
    fit_manifest_temperature,
    read_manifest_predictions,
    select_manifest_policy,
    v5_report,
    validate_artifact,
)


def row(path, label, split, group, family=None, **extra):
    result = {
        "relative_path": path,
        "label": label,
        "media_type": "image",
        "modality": "image",
        "source_dataset": "phone_real" if label == "real" else "generated_faces",
        "source_id": path.rsplit("/", 1)[-1],
        "identity_id": group,
        "identity_group": group,
        "manipulation_type": "none" if label == "real" else "synthesis",
        "generator_family": "real" if label == "real" else (family or "known_gan"),
        "compression_slice": "native",
        "robustness_slice": "clean",
        "split": split,
        "generator_disjoint": "yes" if split == "sealed_test" and label == "fake" else "no",
        "provenance": "custodian metadata",
        "usage_note": "evaluation only",
        "sha256": "",
        "width": "16",
        "height": "16",
        "status": "ok",
        "validation_errors": "",
    }
    result.update(extra)
    return result


def image(root, relative, colour=(30, 50, 80)):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), colour).save(path)
    return path


def complete_rows(root):
    specs = [
        ("cal/real.png", "real", "calibration", "cal-real", None, 0.20),
        ("cal/fake.png", "fake", "calibration", "cal-fake", "known_gan", 0.70),
        ("val/real.png", "real", "validation", "val-real", None, 0.10),
        ("val/fake.png", "fake", "validation", "val-fake", "known_gan", 0.90),
        ("sealed/real.png", "real", "sealed_test", "test-real", None, 0.25),
        ("sealed/fake.png", "fake", "sealed_test", "test-fake", "unseen_generator", 0.75),
    ]
    rows, predictions = [], []
    for number, (path, label, split, group, family, score) in enumerate(specs):
        media = image(root, path, (number + 10, 30, 70))
        item = row(path, label, split, group, family)
        item["sha256"] = hashlib.sha256(media.read_bytes()).hexdigest()
        rows.append(item)
        predictions.append({"path": path, "p_fake": str(score)})
    return rows, predictions


def fixture_files(tmp_path):
    dataset = tmp_path / "data"
    rows, predictions = complete_rows(dataset)
    manifest = tmp_path / "manifest.csv"
    prediction_file = tmp_path / "predictions.csv"
    write_csv(manifest, MANIFEST_FIELDS, rows)
    with prediction_file.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["path", "p_fake"])
        writer.writeheader()
        writer.writerows(predictions)
    return dataset, manifest, prediction_file


def evaluate_command(dataset, manifest, predictions):
    return [
        sys.executable, str(ROOT / "scripts" / "evaluate.py"),
        "--from-csv", str(predictions), "--manifest", str(manifest),
        "--dataset-root", str(dataset), "--model-id", "test-model-v3",
        "--bootstrap", "0",
    ]


def test_manifest_rejects_every_cross_split_leakage_and_unsafe_path():
    rows = [
        row("real/a.png", "real", "train", "same-person"),
        row("real/a.png", "real", "validation", "same-person"),
        row("fake/train.png", "fake", "train", "known-train", "known_gan"),
        row("fake/sealed.png", "fake", "sealed_test", "sealed", "known_gan"),
        row("../outside.png", "fake", "calibration", "safe-cal", "other_gan"),
    ]
    rows[2]["generator_disjoint"] = "yes"
    errors = manifest_validation_errors(rows)

    message = "\n".join(errors)
    assert "file overlap across splits" in message
    assert "identity/original-video group overlap" in message
    assert "generator-family leakage" in message
    assert "unsafe relative_path" in message


def test_manifest_cli_returns_nonzero_with_actionable_errors(tmp_path, monkeypatch):
    dataset = tmp_path / "dataset"
    image(dataset, "sample.png")
    metadata = dataset / "metadata.csv"
    metadata.write_text("relative_path,label\nsample.png,real\n", encoding="utf-8")
    output = tmp_path / "manifest.csv"
    monkeypatch.setattr(sys, "argv", ["dataset_manifest.py", "--dataset", str(dataset),
                                       "--metadata", str(metadata), "--out", str(output)])

    assert manifest_main() == 2
    assert output.is_file(), "invalid rows remain in the audit ledger"


def test_empty_or_schema_incomplete_manifests_fail_validation():
    assert manifest_validation_errors([]) == ["manifest has no records"]
    errors = manifest_validation_errors([{"relative_path": "only-a-path.png"}])
    assert "manifest missing required column: split" in errors
    assert "manifest missing required column: compression_slice" in errors


def test_symlink_media_paths_are_confined_when_supported(tmp_path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    outside = image(tmp_path, "outside.png")
    link = dataset / "escaped.png"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("this Windows test environment does not permit symlink creation")

    _, error = safe_media_path(dataset, "escaped.png")
    assert "symlink" in error


def test_temperature_scaling_uses_only_calibration_and_supports_probability_transform():
    labels = [1, 1, 0, 0]
    raw = [0.99, 0.90, 0.10, 0.01]
    fitted = fit_temperature(labels, probabilities=raw)
    transformed = apply_temperature(probabilities=raw, temperature=fitted["temperature"])

    assert fitted["input"] == "probabilities_logit_transform"
    assert fitted["nll_after"] <= fitted["nll_before"] + 1e-12
    assert all(0 <= value <= 1 for value in transformed)
    assert set(decide_with_abstention([0.2, 0.5, 0.8], 0.3, 0.7)) == {
        "real", "inconclusive", "fake"}


def test_manifest_calibration_and_threshold_policy_never_use_sealed_test(tmp_path):
    rows, predictions = complete_rows(tmp_path / "data")
    manifest = tmp_path / "manifest.csv"
    prediction_file = tmp_path / "predictions.csv"
    write_csv(manifest, MANIFEST_FIELDS, rows)
    with prediction_file.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["path", "p_fake"])
        writer.writeheader()
        writer.writerows(predictions)

    joined = read_manifest_predictions(prediction_file, manifest, tmp_path / "data")
    calibration = fit_manifest_temperature(joined)
    first = select_manifest_policy(joined, target_fpr=0.5, target_fnr=0.5)
    # A sealed-test outlier must be irrelevant to validation threshold choice.
    next(item for item in joined if item["split"] == "sealed_test" and item["label"] == "real")["p_fake"] = "0.999999"
    calibration_again = fit_manifest_temperature(joined)
    second = select_manifest_policy(joined, target_fpr=0.5, target_fnr=0.5)

    assert calibration["fitted_on_split"] == "calibration"
    assert calibration["temperature"] == pytest.approx(calibration_again["temperature"])
    assert first["fitted_on_split"] == "validation"
    assert first["sealed_test_rows_used"] == 0
    assert (first["real_threshold"], first["fake_threshold"]) == (
        second["real_threshold"], second["fake_threshold"])


def test_v5_report_contains_required_metric_and_slice_data(tmp_path):
    rows, predictions = complete_rows(tmp_path / "data")
    manifest = tmp_path / "manifest.csv"
    prediction_file = tmp_path / "predictions.csv"
    write_csv(manifest, MANIFEST_FIELDS, rows)
    with prediction_file.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["path", "p_fake"])
        writer.writeheader()
        writer.writerows(predictions)
    joined = read_manifest_predictions(prediction_file, manifest, tmp_path / "data")
    sealed = [item for item in joined if item["split"] == "sealed_test"]
    report = v5_report(sealed, threshold=0.5, bootstrap=10)

    overall = report["overall"]
    for key in ("precision", "recall", "specificity", "f1", "roc_auc", "pr_auc",
                "fpr", "fnr", "confusion_matrix", "brier_score",
                "expected_calibration_error", "reliability_diagram",
                "bootstrap_confidence_intervals"):
        assert key in overall
    assert "generated_faces" in report["per_dataset"]
    assert "synthesis" in report["per_manipulation"]
    for name in ("jpeg", "resize", "blur", "screenshot", "low_light"):
        assert name in report["per_robustness_slice"]


def test_manifest_reader_refuses_missing_predictions(tmp_path):
    rows, predictions = complete_rows(tmp_path / "data")
    manifest = tmp_path / "manifest.csv"
    prediction_file = tmp_path / "predictions.csv"
    write_csv(manifest, MANIFEST_FIELDS, rows)
    with prediction_file.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["path", "p_fake"])
        writer.writeheader()
        writer.writerows(predictions[:-1])

    with pytest.raises(ValueError, match="missing manifest samples"):
        read_manifest_predictions(prediction_file, manifest, tmp_path / "data")


def test_prediction_logits_require_a_safe_binary_convention(tmp_path):
    dataset, manifest, predictions = fixture_files(tmp_path)
    with predictions.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    unsafe = tmp_path / "unsafe-logits.csv"
    with unsafe.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["path", "p_fake", "fake_logit"])
        writer.writeheader()
        writer.writerows([{**item, "fake_logit": "2.0"} for item in rows])
    with pytest.raises(ValueError, match="lone class logit is unsafe"):
        read_manifest_predictions(unsafe, manifest, dataset)

    safe = tmp_path / "safe-logits.csv"
    with safe.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["path", "p_fake", "fake_logit", "real_logit"])
        writer.writeheader()
        writer.writerows([
            {**item, "fake_logit": "2.0", "real_logit": "0.5"} for item in rows
        ])
    joined = read_manifest_predictions(safe, manifest, dataset)
    assert all(item["binary_logit"] == pytest.approx(1.5) for item in joined)


def test_cli_enforces_three_separate_frozen_artifact_stages(tmp_path):
    dataset, manifest, predictions = fixture_files(tmp_path)
    calibration = tmp_path / "calibration.json"
    thresholds = tmp_path / "thresholds.json"
    report = tmp_path / "sealed-report.json"
    base = evaluate_command(dataset, manifest, predictions)

    fit = subprocess.run(
        base + ["--fit-temperature", "--split", "calibration",
                "--calibration-out", str(calibration)],
        cwd=tmp_path, text=True, capture_output=True, check=False)
    assert fit.returncode == 0, fit.stderr
    fitted = json.loads(calibration.read_text(encoding="utf-8"))
    assert fitted["schema_version"] == ARTIFACT_SCHEMA_VERSION
    assert fitted["artifact_type"] == CALIBRATION_ARTIFACT
    assert fitted["source_split"] == "calibration"
    assert fitted["class_convention"] == CLASS_CONVENTION
    assert fitted["model"] == {"identifier": "test-model-v3", "sha256": None}
    assert len(fitted["manifest_sha256"]) == 64
    assert fitted["created_at"]

    select = subprocess.run(
        base + ["--calibration-in", str(calibration), "--select-thresholds",
                "--split", "validation", "--thresholds-out", str(thresholds)],
        cwd=tmp_path, text=True, capture_output=True, check=False)
    assert select.returncode == 0, select.stderr
    selected = json.loads(thresholds.read_text(encoding="utf-8"))
    assert selected["artifact_type"] == THRESHOLD_ARTIFACT
    assert selected["source_split"] == "validation"
    assert selected["policy"]["source_split"] == "validation"
    assert selected["calibration_artifact_sha256"] == hashlib.sha256(
        calibration.read_bytes()).hexdigest()

    frozen = (calibration.read_bytes(), thresholds.read_bytes())
    sealed = subprocess.run(
        base + ["--calibration-in", str(calibration), "--thresholds-in",
                str(thresholds), "--split", "sealed_test", "--json-report",
                str(report)],
        cwd=tmp_path, text=True, capture_output=True, check=False)
    assert sealed.returncode == 0, sealed.stderr
    assert frozen == (calibration.read_bytes(), thresholds.read_bytes())
    result = json.loads(report.read_text(encoding="utf-8"))
    assert result["evaluated_split"] == "sealed_test"
    assert result["protocol"]["calibrated"] is True
    assert result["protocol"]["positive_class"] == "fake"
    assert result["decision_policy"]["source_split"] == "validation"
    assert "three_way_decisions" in result


def test_fit_temperature_with_sealed_test_split_fails_loudly(tmp_path):
    dataset, manifest, predictions = fixture_files(tmp_path)
    blocked = tmp_path / "must-not-exist.json"
    result = subprocess.run(
        evaluate_command(dataset, manifest, predictions)
        + ["--fit-temperature", "--split", "sealed_test",
           "--calibration-out", str(blocked)],
        cwd=tmp_path, text=True, capture_output=True, check=False)

    assert result.returncode != 0
    assert "--fit-temperature requires --split calibration" in result.stderr
    assert not blocked.exists()


def test_sealed_test_cannot_select_or_overwrite_frozen_artifacts(tmp_path):
    dataset, manifest, predictions = fixture_files(tmp_path)
    calibration = tmp_path / "calibration.json"
    thresholds = tmp_path / "thresholds.json"
    base = evaluate_command(dataset, manifest, predictions)
    assert subprocess.run(
        base + ["--fit-temperature", "--split", "calibration",
                "--calibration-out", str(calibration)],
        cwd=tmp_path, capture_output=True, check=False).returncode == 0
    assert subprocess.run(
        base + ["--calibration-in", str(calibration), "--select-thresholds",
                "--split", "validation", "--thresholds-out", str(thresholds)],
        cwd=tmp_path, capture_output=True, check=False).returncode == 0
    frozen = (calibration.read_bytes(), thresholds.read_bytes())

    selection = subprocess.run(
        base + ["--calibration-in", str(calibration), "--select-thresholds",
                "--split", "sealed_test", "--thresholds-out", str(tmp_path / "new.json")],
        cwd=tmp_path, text=True, capture_output=True, check=False)
    overwrite = subprocess.run(
        base + ["--calibration-in", str(calibration), "--thresholds-in",
                str(thresholds), "--split", "sealed_test", "--json-report",
                str(calibration)],
        cwd=tmp_path, text=True, capture_output=True, check=False)

    assert selection.returncode != 0
    assert "--select-thresholds requires --split validation" in selection.stderr
    assert overwrite.returncode != 0
    assert "must not overwrite" in overwrite.stderr
    assert frozen == (calibration.read_bytes(), thresholds.read_bytes())


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema_version", 999, "schema_version"),
        ("source_split", "sealed_test", "source split"),
        ("class_convention", {"positive_class": "real"}, "class convention"),
        ("manifest_sha256", "0" * 64, "manifest hash"),
        ("model", {"identifier": "another-model", "sha256": None}, "model identifier"),
    ],
)
def test_incompatible_calibration_artifacts_are_rejected(
        tmp_path, field, value, message):
    _dataset, manifest, _predictions = fixture_files(tmp_path)
    context = artifact_context(manifest, model_id="test-model-v3")
    record = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "artifact_type": CALIBRATION_ARTIFACT,
        "created_at": "2026-08-20T00:00:00+00:00",
        "source_split": "calibration",
        "class_convention": dict(CLASS_CONVENTION),
        "model": dict(context["model"]),
        "manifest_sha256": context["manifest_sha256"],
        "calibration": {"method": "temperature_scaling", "temperature": 1.0},
    }
    record[field] = value

    with pytest.raises(ValueError, match=message):
        validate_artifact(record, CALIBRATION_ARTIFACT, context)


def test_leakage_errors_name_both_conflicting_records():
    digest = "a" * 64
    records = [
        row("derived/original.mp4", "real", "train", "original-video-17",
            sha256=digest, modality="video", media_type="video"),
        row("derived/frame-001.png", "fake", "sealed_test", "original-video-17",
            "unseen", sha256="b" * 64),
        row("renamed/copy.png", "real", "validation", "different-person",
            sha256=digest),
    ]
    errors = "\n".join(manifest_validation_errors(records))

    assert "content-hash overlap across splits" in errors
    assert "row 2 (train)" in errors and "row 4 (validation)" in errors
    assert "identity/original-video group overlap" in errors
    assert "row 2 (train)" in errors and "row 3 (sealed_test)" in errors


def test_same_bytes_under_different_names_are_recomputed_and_rejected(tmp_path):
    dataset = tmp_path / "dataset"
    first = image(dataset, "train/original.png")
    second = dataset / "validation/renamed.png"
    second.parent.mkdir(parents=True)
    second.write_bytes(first.read_bytes())
    digest = hashlib.sha256(first.read_bytes()).hexdigest()
    records = [
        row("train/original.png", "real", "train", "group-a", sha256=digest),
        row("validation/renamed.png", "real", "validation", "group-b",
            sha256=digest),
    ]

    errors = "\n".join(manifest_validation_errors(records, dataset))
    assert "content-hash overlap across splits" in errors
    assert "row 2 (train)" in errors and "row 3 (validation)" in errors


@pytest.mark.parametrize("unsafe", [
    "../outside.png", "/etc/passwd", "C:/Windows/win.ini",
    "C:\\Windows\\win.ini", "//server/share/file.png", "a/./b.png",
])
def test_paths_are_portably_relative_on_windows_and_linux(unsafe):
    assert relative_path_error(unsafe)


def test_symlink_escape_rule_is_tested_without_platform_privileges(tmp_path, monkeypatch):
    dataset = tmp_path / "dataset"
    (dataset / "linked").mkdir(parents=True)
    original = Path.is_symlink

    def pretend_symlink(path):
        return path.name == "linked" or original(path)

    monkeypatch.setattr(Path, "is_symlink", pretend_symlink)
    _path, error = safe_media_path(dataset, "linked/file.png")
    assert "symlink" in error


def test_fake_positive_metric_orientation_and_hand_calculated_values():
    labels = [1, 1, 1, 0, 0, 0]
    scores = [0.9, 0.4, 0.8, 0.7, 0.2, 0.1]
    matrix = confusion(labels, scores, 0.5)
    metrics = evaluate(labels, scores, 0.5)

    assert matrix == {"tp": 2, "fp": 1, "tn": 2, "fn": 1}
    for name in ("precision", "recall", "specificity", "f1"):
        assert metrics[name] == pytest.approx(2 / 3)
    assert metrics["fpr"] == pytest.approx(1 / 3)
    assert metrics["fnr"] == pytest.approx(1 / 3)
    assert roc_auc(labels, scores) == pytest.approx(8 / 9)
    assert pr_auc(labels, scores) == pytest.approx(11 / 12)
    assert brier(labels, scores) == pytest.approx(0.95 / 6)
    assert ece(labels, scores, bins=2) == pytest.approx(7 / 60)
    diagram = reliability(labels, scores, bins=2)
    assert [item["n"] for item in diagram] == [3, 3]


def test_undefined_metrics_remain_null_and_threshold_budgets_include_boundaries():
    only_real = evaluate([0, 0], [0.1, 0.2], 0.5)
    only_fake = evaluate([1, 1], [0.8, 0.9], 0.5)
    assert only_real["precision"] is None
    assert only_real["recall"] is None
    assert only_real["f1"] is None
    assert only_real["roc_auc"] is None
    assert only_fake["specificity"] is None
    assert only_fake["fpr"] is None

    with pytest.raises(ValueError, match="FNR budget"):
        select_abstention_thresholds(
            [1, 1, 0, 0], [0.0, 0.9, 0.1, 0.2], target_fnr=0.0)


def test_bootstrap_resamples_independent_video_groups_not_frames():
    labels = [0] * 50 + [1] * 50
    scores = [0.1] * 50 + [0.9] * 50
    groups = ["real-video"] * 50 + ["fake-video"] * 50
    intervals = bootstrap_confidence_intervals(
        labels, scores, groups=groups, resamples=20, seed=7)

    assert intervals["unit"] == "groups"
    assert intervals["independent_units"] == 2
    assert intervals["resamples"] == 20


def test_production_numerical_outputs_are_unchanged_by_score_metadata(
        tmp_path, monkeypatch):
    import inference

    source = image(tmp_path, "fixed.png")

    class FixedEngine:
        classes = ("fake", "real")

        @staticmethod
        def _detect_faces(pil_image):
            return [{"crop": pil_image.copy(), "landmarks": None, "box": (1, 2, 8, 8),
                     "origin": (0, 0), "frame": pil_image.size, "found": True}]

        @staticmethod
        def _probs_raw(_crop):
            return np.asarray([0.73, 0.27])

        @staticmethod
        def explain(_face, _landmarks):
            return None

    monkeypatch.setattr(inference, "_get_engine", lambda: FixedEngine())
    monkeypatch.setattr(inference, "_get_hf_engines", list)
    result = inference.analyze_file(str(source), "image")

    # These are the pre-V5 numerical/decision fields. New fields are additive.
    assert {key: result[key] for key in (
        "prediction", "confidence", "framesAnalyzed", "faceFound", "facesFound",
    )} == {
        "prediction": "deepfake", "confidence": 73, "framesAnalyzed": 1,
        "faceFound": True, "facesFound": 1,
    }
    assert result["ensemble"][0]["pFake"] == 0.73
    assert result["uncalibratedScore"] == 0.73
    assert result["scoreLabel"] == "uncalibrated model score"
    assert result["scoreCalibrated"] is False
