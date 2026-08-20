"""Score a labelled image set with the live engine and report every metric.

    python scripts/evaluate.py                      # scores eval_data/
    python scripts/evaluate.py --data DIR --seen ffhq,sg1,tpdn,diffusion
    python scripts/evaluate.py --from-csv preds.csv # recompute, no model needed
    python scripts/evaluate.py --conditions SRC --out eval_data/real/processed

Expected layout — the folder under each class names the source, and the
source is what the per-source table reports on:

    eval_data/
      real/ffhq/*.jpg          real/phone/*.jpg      real/screenshot/*.png
      fake/stylegan/*.jpg      fake/dfdc/*.jpg       fake/diffusion/*.jpg

Two things this deliberately does:

**It scores through `inference.score_image`**, the same preprocessing a
real upload gets. A benchmark that runs its own resize pipeline measures a
model the users never meet.

**It computes nothing itself.** Every number comes from `ds_metrics`, so
the figures printed here and the figures a Kaggle notebook produces are
the same arithmetic — the notebook writes a predictions CSV, and
`--from-csv` turns it into the report.

Images only. Video is scored frame-by-frame by the app; a per-frame
benchmark needs a different labelling scheme than this one.
"""
import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "backend"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import ds_metrics as M
from dataset_common import (
    ROBUSTNESS_SLICES,
    ManifestValidationError,
    normalized_path,
    relative_path_error,
    validate_manifest,
)
from dataset_common import read_csv as read_manifest_csv

IMAGE_EXT = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
FIELDS = ["path", "source", "label", "y_true", "p_fake", "logit_fake", "group"]
ARTIFACT_SCHEMA_VERSION = 1
CALIBRATION_ARTIFACT = "deepshield.temperature_calibration"
THRESHOLD_ARTIFACT = "deepshield.abstention_thresholds"
CLASS_CONVENTION = {
    "positive_class": "fake",
    "negative_class": "real",
    "score_field": "p_fake",
    "higher_score_means": "more likely fake",
}


# --------------------------------------------------------------- collecting

def group_of(path):
    """Which images must never be split across train and test.

    Defaults to the filename stem with any `__cond-*` suffix removed, so
    the four processed variants of one photograph count as one source
    image rather than four independent samples. A `groups.csv`
    (path,group) next to the data overrides this — DFDC needs it, because
    every fake made from the same original video shares that original."""
    stem = os.path.splitext(os.path.basename(path))[0]
    return stem.split("__cond-")[0]


def load_group_overrides(data_dir):
    path = os.path.join(data_dir, "groups.csv")
    if not os.path.exists(path):
        return {}
    with open(path, newline="", encoding="utf-8") as f:
        return {os.path.normcase(r["path"]): r["group"]
                for r in csv.DictReader(f) if r.get("path")}


def collect(data_dir):
    """→ [(path, source, label, y_true)] walking real/<source> and fake/<source>."""
    items = []
    for label, y_true in (("real", 0), ("fake", 1)):
        root = os.path.join(data_dir, label)
        if not os.path.isdir(root):
            continue
        for dirpath, _, filenames in os.walk(root):
            rel = os.path.relpath(dirpath, root)
            source = "(root)" if rel == "." else rel.replace(os.sep, "/").split("/")[0]
            for name in sorted(filenames):
                if name.lower().endswith(IMAGE_EXT):
                    items.append((os.path.join(dirpath, name), source, label, y_true))
    return items


# ----------------------------------------------------------------- scoring

def score_all(items, overrides, out_csv, limit=None):
    import inference

    if not inference.engine_available():
        sys.exit("no model available - nothing to evaluate")
    if limit:
        items = items[:limit]

    rows, failures = [], []
    started = time.time()
    for i, (path, source, label, y_true) in enumerate(items, 1):
        try:
            p_fake = inference.score_image(path)
        except Exception as exc:  # noqa: BLE001
            failures.append((path, f"{type(exc).__name__}: {exc}"))
            continue
        rows.append({
            "path": os.path.relpath(path, ROOT).replace(os.sep, "/"),
            "source": source, "label": label, "y_true": y_true,
            "p_fake": f"{p_fake:.6f}",
            "group": overrides.get(os.path.normcase(path)) or group_of(path),
        })
        if i % 25 == 0 or i == len(items):
            rate = i / max(time.time() - started, 1e-6)
            print(f"\r  scored {i}/{len(items)}  ({rate:.1f} img/s)", end="", flush=True)
    print()

    if out_csv:
        os.makedirs(os.path.dirname(os.path.abspath(out_csv)) or ".", exist_ok=True)
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(rows)
        print(f"  predictions -> {os.path.relpath(out_csv, ROOT)}")

    if failures:
        print(f"\n  {len(failures)} image(s) could not be scored:")
        for path, why in failures[:10]:
            print(f"    {os.path.basename(path)}: {why}")
    return rows


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    missing = [c for c in ("y_true", "p_fake") if rows and c not in rows[0]]
    if missing:
        sys.exit(f"{path} is missing column(s): {', '.join(missing)}")
    for r in rows:
        r.setdefault("source", "(all)")
        r.setdefault("label", "fake" if int(r["y_true"]) else "real")
    return rows


# --------------------------------------------------------- V5 manifest path

def _prediction_key(row):
    return normalized_path(row.get("path") or row.get("file") or row.get("relative_path") or "")


def _binary_logit(row, number):
    direct_values = [
        row.get("logit_fake_minus_real", ""),
        row.get("log_odds_fake", ""),
    ]
    direct = [value for value in direct_values if str(value).strip()]
    fake = row.get("fake_logit", "") or row.get("logit_fake", "")
    real = row.get("real_logit", "")
    if direct and (str(fake).strip() or str(real).strip()):
        raise ValueError(
            f"predictions row {number}: provide either fake-minus-real logit "
            "or both class logits, not both forms")
    if len(direct) > 1:
        raise ValueError(
            f"predictions row {number}: duplicate fake-minus-real logit columns")
    if direct:
        value = direct[0]
    elif str(fake).strip() or str(real).strip():
        if not str(fake).strip() or not str(real).strip():
            raise ValueError(
                f"predictions row {number}: a lone class logit is unsafe; "
                "provide both fake_logit and real_logit")
        try:
            value = float(fake) - float(real)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"predictions row {number}: class logits must be numeric") from exc
    else:
        return ""
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"predictions row {number}: fake-minus-real logit must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(
            f"predictions row {number}: fake-minus-real logit must be finite")
    return result


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact_context(manifest_path, *, model_id=None, model_artifact=None):
    """Bind generated artifacts to one manifest and one scored model."""
    manifest = Path(manifest_path)
    if not manifest.is_file():
        raise ValueError(f"manifest file not found: {manifest}")
    model_path = Path(model_artifact) if model_artifact else None
    if model_path is not None and not model_path.is_file():
        raise ValueError(f"model artifact not found: {model_path}")
    identifier = str(model_id or "").strip()
    if not identifier and model_path is None:
        raise ValueError("provide --model-id or --model-artifact to bind V5 artifacts")
    return {
        "manifest_sha256": _sha256_file(manifest),
        "model": {
            "identifier": identifier or model_path.name,
            "sha256": _sha256_file(model_path) if model_path is not None else None,
        },
    }


def _artifact_header(artifact_type, source_split, context):
    return {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "artifact_type": artifact_type,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_split": source_split,
        "class_convention": dict(CLASS_CONVENTION),
        "model": dict(context["model"]),
        "manifest_sha256": context["manifest_sha256"],
    }


def calibration_artifact(record, context):
    return {
        **_artifact_header(CALIBRATION_ARTIFACT, "calibration", context),
        "calibration": dict(record),
    }


def threshold_artifact(policy, context, calibration_sha256):
    return {
        **_artifact_header(THRESHOLD_ARTIFACT, "validation", context),
        "calibration_artifact_sha256": calibration_sha256,
        "policy": dict(policy),
    }


def _validate_created_at(value):
    try:
        created = datetime.fromisoformat(str(value))
    except (TypeError, ValueError) as exc:
        raise ValueError("artifact creation timestamp is missing or invalid") from exc
    if created.tzinfo is None:
        raise ValueError("artifact creation timestamp must include a timezone")


def validate_artifact(record, expected_type, context, *, calibration_sha256=None):
    """Reject stale, foreign, malformed, or scientifically incompatible artifacts."""
    if not isinstance(record, dict):
        raise TypeError("artifact must be a JSON object")
    if record.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise ValueError(
            f"incompatible artifact schema_version: {record.get('schema_version')!r}")
    if record.get("artifact_type") != expected_type:
        raise ValueError(
            f"incompatible artifact type: expected {expected_type!r}, "
            f"got {record.get('artifact_type')!r}")
    expected_split = "calibration" if expected_type == CALIBRATION_ARTIFACT else "validation"
    if record.get("source_split") != expected_split:
        raise ValueError(
            f"incompatible artifact source split: expected {expected_split!r}, "
            f"got {record.get('source_split')!r}")
    if record.get("class_convention") != CLASS_CONVENTION:
        raise ValueError("incompatible artifact class convention (fake must be positive)")
    if record.get("manifest_sha256") != context["manifest_sha256"]:
        raise ValueError("artifact manifest hash does not match the current manifest")
    _validate_created_at(record.get("created_at"))

    stored_model = record.get("model")
    if not isinstance(stored_model, dict):
        raise TypeError("artifact model identity is missing")
    current_model = context["model"]
    if stored_model.get("identifier") != current_model.get("identifier"):
        raise ValueError("artifact model identifier does not match the current model")
    stored_hash = stored_model.get("sha256")
    if stored_hash is not None and stored_hash != current_model.get("sha256"):
        raise ValueError("artifact model SHA-256 does not match the current model")
    if stored_hash is None and current_model.get("sha256") is not None:
        raise ValueError("artifact lacks the model SHA-256 required by this run")

    if expected_type == CALIBRATION_ARTIFACT:
        payload = record.get("calibration")
        if not isinstance(payload, dict) or payload.get("method") != "temperature_scaling":
            raise ValueError("calibration artifact has no temperature-scaling payload")
        try:
            temperature = float(payload["temperature"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("calibration artifact temperature is missing or invalid") from exc
        if not 0 < temperature < float("inf"):
            raise ValueError("calibration artifact temperature must be positive and finite")
    else:
        if record.get("calibration_artifact_sha256") != calibration_sha256:
            raise ValueError("threshold artifact was selected with a different calibration artifact")
        payload = record.get("policy")
        if not isinstance(payload, dict) or payload.get("source_split") != "validation":
            raise ValueError("threshold artifact has no validation-only policy")
        try:
            real = float(payload["real_threshold"])
            fake = float(payload["fake_threshold"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("threshold artifact boundaries are missing or invalid") from exc
        if not 0 <= real < fake <= 1:
            raise ValueError("threshold artifact must satisfy 0 <= real < fake <= 1")
    return record


def load_artifact(path, expected_type, context, *, calibration_sha256=None):
    with open(path, encoding="utf-8") as stream:
        record = json.load(stream)
    return validate_artifact(
        record, expected_type, context, calibration_sha256=calibration_sha256)


def write_artifact(path, record):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(record, stream, indent=2, sort_keys=True)
        stream.write("\n")


def read_manifest_predictions(predictions_path, manifest_path, dataset_root=None):
    """Join model output to an already validated V5 manifest.

    Labels, split assignments, provenance and robustness slices come only
    from the manifest. A prediction CSV cannot relabel a sample or quietly
    substitute a path, which is how an evaluation protocol becomes a report
    generator rather than an experiment record.
    """
    manifest = read_manifest_csv(Path(manifest_path))
    if not manifest:
        raise ManifestValidationError(f"manifest is empty: {manifest_path}")
    validate_manifest(manifest, Path(dataset_root) if dataset_root else None)
    with open(predictions_path, newline="", encoding="utf-8-sig") as stream:
        predictions = list(csv.DictReader(stream))
    if predictions and "p_fake" not in predictions[0]:
        raise ValueError(f"predictions CSV is missing required p_fake column: {predictions_path}")
    by_path = {}
    for number, row in enumerate(predictions, start=2):
        raw_path = row.get("path") or row.get("file") or row.get("relative_path") or ""
        path_error = relative_path_error(raw_path)
        if path_error:
            raise ValueError(f"predictions row {number}: {path_error}")
        key = _prediction_key(row)
        if not key:
            raise ValueError(f"predictions row {number}: missing path")
        if key in by_path:
            raise ValueError(f"predictions row {number}: duplicate prediction path {key!r}")
        try:
            score = float(row["p_fake"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"predictions row {number}: p_fake must be numeric") from exc
        if not 0 <= score <= 1:
            raise ValueError(f"predictions row {number}: p_fake must be in [0, 1]")
        by_path[key] = {**row, "binary_logit": _binary_logit(row, number)}

    joined, missing = [], []
    for number, meta in enumerate(manifest, start=2):
        key = normalized_path(meta["relative_path"])
        prediction = by_path.get(key)
        if prediction is None:
            missing.append(f"row {number} ({meta['split']}): {key}")
            continue
        label = meta["label"].lower()
        joined.append({
            **meta,
            "path": key,
            "source": meta["source_dataset"],
            "label": label,
            "y_true": "1" if label == "fake" else "0",
            "raw_p_fake": prediction["p_fake"],
            "p_fake": prediction["p_fake"],
            "group": meta.get("identity_group") or meta.get("identity_id"),
            "binary_logit": prediction.get("binary_logit", ""),
        })
    if missing:
        shown = "\n  - ".join(missing[:20])
        more = f"\n  - ... and {len(missing) - 20} more" if len(missing) > 20 else ""
        raise ValueError("predictions are missing manifest samples:\n  - " + shown + more)
    extra = sorted(set(by_path) - {normalized_path(r["relative_path"]) for r in manifest})
    if extra:
        shown = ", ".join(extra[:10])
        raise ValueError(f"predictions contain paths absent from manifest: {shown}")
    return joined


def _rows_for_split(rows, split):
    if split == "all":
        return list(rows)
    selected = [row for row in rows if row.get("split") == split]
    if not selected:
        raise ValueError(f"manifest contains no rows in split {split!r}")
    return selected


def fit_manifest_temperature(rows):
    """Fit only on calibration rows; do not transform any other split."""
    calibration_rows = _rows_for_split(rows, "calibration")
    labels, probabilities = split(calibration_rows)
    logit_values = [row.get("binary_logit", "") for row in calibration_rows]
    if all(str(value).strip() for value in logit_values):
        record = M.fit_temperature(labels, logits=[float(value) for value in logit_values])
    else:
        record = M.fit_temperature(labels, probabilities=probabilities)
    record["fitted_on_split"] = "calibration"
    record["thresholds_fitted_on_split"] = None
    return record


def apply_manifest_temperature(rows, record):
    """Apply a saved calibration record; fitting is deliberately impossible here."""
    record = record.get("calibration", record)
    if record.get("method") != "temperature_scaling":
        raise ValueError("calibration record is not a temperature-scaling record")
    if record.get("fitted_on_split") != "calibration":
        raise ValueError("calibration record was not fitted on the calibration split")
    temperature = float(record["temperature"])
    for row in rows:
        value = row.get("binary_logit", "")
        calibrated = (M.apply_temperature(logits=[float(value)], temperature=temperature)[0]
                      if str(value).strip() else
                      M.apply_temperature(probabilities=[float(row["p_fake"])],
                                          temperature=temperature)[0])
        row["p_fake"] = f"{calibrated:.12g}"
    return record


def select_manifest_policy(rows, target_fpr=0.01, target_fnr=0.01,
                           minimum_band=0.10):
    """Select real/fake/inconclusive thresholds from validation rows only."""
    validation_rows = _rows_for_split(rows, "validation")
    y, scores = split(validation_rows)
    policy = M.select_abstention_thresholds(
        y, scores, target_fpr=target_fpr, target_fnr=target_fnr,
        minimum_band=minimum_band)
    policy["fitted_on_split"] = "validation"
    policy["sealed_test_rows_used"] = 0
    return policy


def _metric_block(rows, threshold, bootstrap=0):
    y, scores = split(rows)
    metrics = M.evaluate(y, scores, threshold)
    metrics.update({
        "brier_score": M.brier(y, scores),
        "expected_calibration_error": M.ece(y, scores),
        "reliability_diagram": M.reliability(y, scores),
        "confusion_matrix": {"tp": metrics["tp"], "fp": metrics["fp"],
                             "tn": metrics["tn"], "fn": metrics["fn"]},
    })
    metrics["bootstrap_confidence_intervals"] = M.bootstrap_confidence_intervals(
        y, scores, threshold=threshold, resamples=bootstrap,
        groups=[row.get("identity_group") or row.get("identity_id") or row.get("group")
                for row in rows])
    return metrics


def _group_metrics(rows, field, threshold, bootstrap):
    grouped = {}
    values = sorted({str(row.get(field, "") or "unknown") for row in rows})
    for value in values:
        grouped[value] = _metric_block(
            [row for row in rows if str(row.get(field, "") or "unknown") == value],
            threshold, bootstrap)
    return grouped


def v5_report(rows, threshold, bootstrap=200, decision_policy=None, *, calibrated=False):
    """JSON-ready result with calibrated-score and robustness evidence."""
    report_rows = list(rows)
    if not report_rows:
        raise ValueError("no rows selected for V5 evaluation")
    slices = _group_metrics(report_rows, "robustness_slice", threshold, bootstrap)
    for name in ROBUSTNESS_SLICES:
        slices.setdefault(name, {"n": 0, "not_measured": True})
    result = {
        "protocol": {
            "positive_class": "fake",
            "score": ("temperature-calibrated fake-class score" if calibrated
                      else "uncalibrated model score"),
            "calibrated": bool(calibrated),
            "threshold_source": ("validation" if decision_policy else "fixed CLI value"),
            "bootstrap_resamples": int(bootstrap),
            "bootstrap_unit": "identity/original-video group",
        },
        "overall": _metric_block(report_rows, threshold, bootstrap),
        "per_dataset": _group_metrics(report_rows, "source_dataset", threshold, bootstrap),
        "per_manipulation": _group_metrics(report_rows, "manipulation_type", threshold, bootstrap),
        "per_robustness_slice": slices,
        "decision_policy": decision_policy,
    }
    if decision_policy:
        labels, scores = split(report_rows)
        result["three_way_decisions"] = M.abstention_summary(
            labels, scores, decision_policy["real_threshold"],
            decision_policy["fake_threshold"])
    return result


def print_v5_summary(report, evaluated_split):
    m = report["overall"]
    print("\n" + "=" * 70)
    print(f"V5 EVALUATION — {evaluated_split}")
    print("=" * 70)
    print(M.format_report(m))
    print(f"\n  Brier score: {m['brier_score']:.6f}" if m["brier_score"] is not None
          else "\n  Brier score: n/a")
    print("  Expected calibration error: " +
          (f"{m['expected_calibration_error']:.6f}" if m["expected_calibration_error"] is not None else "n/a"))
    print("  Reliability-diagram data and bootstrap confidence intervals are in --json-report.")
    if report.get("decision_policy"):
        p = report["decision_policy"]
        print(f"  Validation-only abstention policy: real <= {p['real_threshold']:.3f}; "
              f"fake >= {p['fake_threshold']:.3f}; otherwise inconclusive.")


# ------------------------------------------------------------------ report

def split(rows):
    return [int(r["y_true"]) for r in rows], [float(r["p_fake"]) for r in rows]


def per_source_table(rows, threshold):
    """The dataset matrix, one line per source.

    A real source can only be judged on false positives and a fake source
    only on detection, so the last column changes meaning by class and
    says which it is."""
    sources = {}
    for r in rows:
        sources.setdefault((r["label"], r["source"]), []).append(r)

    lines = ["    class  source                 n    mean P(fake)   verdict",
             "    " + "-" * 66]
    for (label, source), group in sorted(sources.items()):
        y, s = split(group)
        m = M.evaluate(y, s, threshold)
        mean_p = sum(s) / len(s)
        if label == "real":
            rate = m["fpr"]
            verdict = f"{rate * 100:6.2f}% called fake" if rate is not None else "n/a"
        else:
            rate = m["recall"]
            verdict = f"{rate * 100:6.2f}% detected" if rate is not None else "n/a"
        lines.append(f"    {label:5s}  {source:20s} {len(group):5d}"
                     f"      {mean_p:.3f}      {verdict}")
    return "\n".join(lines)


def sweep_table(rows):
    y, s = split(rows)
    lines = ["    threshold   accuracy   recall    FPR       F1",
             "    " + "-" * 48]
    for m in M.sweep(y, s):
        def pct(v):
            return "   n/a  " if v is None else f"{v * 100:6.2f}%"
        lines.append(f"      {m['threshold']:.2f}      {pct(m['accuracy'])}  "
                     f"{pct(m['recall'])}  {pct(m['fpr'])}  {pct(m['f1'])}")
    return "\n".join(lines)


def group_note(rows):
    """Repeated groups mean correlated samples: a set of 500 crops from 50
    videos is closer to 50 independent tests than 500, and saying so keeps
    the headline number honest."""
    groups = {r.get("group") for r in rows if r.get("group")}
    if not groups or len(groups) == len(rows):
        return ""
    return (f"\n  {len(rows)} images come from {len(groups)} independent "
            f"groups (person / video / source).\n  Treat the sample size as "
            f"{len(groups)}, not {len(rows)} — samples inside a group are "
            "correlated.")


def report(rows, threshold, seen=None, target_fpr=0.01):
    if not rows:
        sys.exit("no predictions to report on")

    y, s = split(rows)
    print("\n" + "=" * 70)
    print("OVERALL")
    print("=" * 70)
    print(M.format_report(M.evaluate(y, s, threshold)))
    print(group_note(rows))

    print("\n" + "=" * 70)
    print("BY SOURCE  - the dataset matrix")
    print("=" * 70)
    print(per_source_table(rows, threshold))

    print("\n" + "=" * 70)
    print("THRESHOLD SWEEP  - 0.5 is a convention, not a result")
    print("=" * 70)
    print(sweep_table(rows))

    calibration(rows)

    point = M.threshold_for_fpr(y, s, target_fpr)
    if point:
        t, fpr, recall = point
        print(f"\n  For a false-positive rate at or below {target_fpr * 100:g}%:")
        print(f"    threshold {t:.3f}  ->  FPR {fpr * 100:.2f}%  "
              f"detection {recall * 100:.2f}%")
    else:
        print(f"\n  No threshold reaches an FPR of {target_fpr * 100:g}% "
              "(or one class is missing).")

    if seen:
        cross_dataset(rows, threshold, seen)


def calibration(rows):
    """Whether the percentage the UI prints means anything.

    Two questions, deliberately both asked. The first is about the raw
    probability; the second is about the number a user is actually shown,
    and it is the one that decides whether a certainty band deserves its
    name."""
    y, s = split(rows)

    print("\n" + "=" * 70)
    print("CALIBRATION  - is the number a frequency or just a ranking?")
    print("=" * 70)
    print(M.format_calibration(y, s, mode="positive"))
    print()
    print(M.format_calibration(y, s, mode="confidence"))

    try:
        sys.path.insert(0, os.path.join(ROOT, "backend"))
        from config import CFG
    except ImportError:
        return

    print("\n" + "=" * 70)
    print("CERTAINTY BANDS  - the labels against what actually happened")
    print("=" * 70)
    print(M.format_bands(M.band_accuracy(y, s, CFG.CERTAINTY_BANDS)))
    print("\n  A band whose observed accuracy is far from its name is a band")
    print("  whose cut point is wrong. These are the numbers that should")
    print("  replace CERTAINTY_BANDS in backend/config.py.")


def cross_dataset(rows, threshold, seen):
    """In-domain against held-out generators.

    The gap between these two blocks is the honest headline. A detector
    that scores 99% on the generators it trained against and 60% on one it
    has never seen is a 60% detector as far as the real world is
    concerned."""
    seen = {x.strip().lower() for x in seen if x.strip()}
    in_domain = [r for r in rows if r["source"].lower() in seen]
    unseen = [r for r in rows if r["source"].lower() not in seen]

    print("\n" + "=" * 70)
    print("CROSS-DATASET  - trained-on vs never-seen")
    print("=" * 70)
    if not in_domain or not unseen:
        print("    Needs both: sources named in --seen, and sources outside it.")
        print(f"    in-domain {len(in_domain)} images, unseen {len(unseen)}.")
        return

    for title, subset in (("IN-DOMAIN  " + ", ".join(sorted(seen)), in_domain),
                          ("UNSEEN  " + ", ".join(sorted(
                              {r['source'] for r in unseen})), unseen)):
        y, s = split(subset)
        print()
        print(M.format_report(M.evaluate(y, s, threshold), title))

    a = M.evaluate(*split(in_domain), threshold)
    b = M.evaluate(*split(unseen), threshold)
    if a["accuracy"] is not None and b["accuracy"] is not None:
        print(f"\n  Generalisation gap: {(a['accuracy'] - b['accuracy']) * 100:+.2f} "
              "points of accuracy")
    if a["recall"] is not None and b["recall"] is not None:
        print(f"  Detection gap:      {(a['recall'] - b['recall']) * 100:+.2f} "
              "points of recall")


# ------------------------------------------------------- condition variants

CONDITIONS = {
    # name          long side   save as   quality
    "orig":        (None,       "JPEG",   95),
    "phone":       (1440,       "JPEG",   92),
    "screenshot":  (1080,       "PNG",    None),
    "social":      (720,        "JPEG",   60),
    "reencode":    (None,       "JPEG",   40),
    # V5 names match the manifest's robustness_slice vocabulary. Keep the
    # older phone/social variants above because existing reports use them.
    "jpeg":        (None,       "JPEG",   60),
    "resize":      (720,        "JPEG",   88),
    "blur":        (None,       "JPEG",   88),
    "low_light":   (None,       "JPEG",   88),
}


def make_conditions(src_dir, out_dir):
    """Write processed variants of every image in src_dir.

    These approximate what happens to a photograph on its way through a
    phone, a screenshot, a messaging app and repeated forwarding. They are
    stand-ins with plausible parameters, not measurements of any specific
    platform's pipeline — the point is to show how the verdict moves as
    evidence is destroyed, which is where a detector's real false-positive
    rate lives."""
    from PIL import Image

    files = [os.path.join(src_dir, f) for f in sorted(os.listdir(src_dir))
             if f.lower().endswith(IMAGE_EXT)]
    if not files:
        sys.exit(f"no images in {src_dir}")

    made = 0
    for path in files:
        stem = os.path.splitext(os.path.basename(path))[0]
        for name, (side, fmt, quality) in CONDITIONS.items():
            target = os.path.join(out_dir, name,
                                  f"{stem}__cond-{name}.{'png' if fmt == 'PNG' else 'jpg'}")
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with Image.open(path) as im:
                im = im.convert("RGB")
                if side and max(im.size) > side:
                    scale = side / max(im.size)
                    im = im.resize((max(1, round(im.width * scale)),
                                    max(1, round(im.height * scale))),
                                   Image.LANCZOS)
                if name == "reencode":                    # forwarded twice
                    import io
                    buf = io.BytesIO()
                    im.save(buf, "JPEG", quality=55)
                    buf.seek(0)
                    im = Image.open(buf).convert("RGB")
                elif name == "blur":
                    from PIL import ImageFilter
                    im = im.filter(ImageFilter.GaussianBlur(radius=1.5))
                elif name == "low_light":
                    from PIL import ImageEnhance
                    im = ImageEnhance.Brightness(im).enhance(0.45)
                im.save(target, fmt, **({"quality": quality} if quality else {}))
            made += 1

    print(f"  {made} variants of {len(files)} image(s) -> {out_dir}")
    print(f"  conditions: {', '.join(CONDITIONS)}")
    print("  Variants of one photo share a group, so they count as one sample.")


# ------------------------------------------------------------ markdown report

def markdown_report(rows, threshold, seen=None, latency=None):
    """The benchmark table, generated rather than typed.

    A field with no data behind it says so and names what would produce it.
    Every headline number in this project is meant to be reproducible from
    the CSV that generated it, and a hand-written table breaks that the
    first time someone forgets to update it.
    """
    y, s = split(rows)
    m = M.evaluate(y, s, threshold)
    groups = {r.get("group") for r in rows if r.get("group")}
    n_groups = len(groups) if groups else len(rows)

    def pct(v):
        return "*not measurable*" if v is None else f"**{v * 100:.2f}%**"

    def num(v, places=4):
        return "*not measurable*" if v is None else f"**{v:.{places}f}**"

    lines = [
        "| Metric | Value | Basis |",
        "|---|---|---|",
        f"| Images scored | **{m['n']:,}** | {m['n_real']} real, {m['n_fake']} fake |",
        (f"| Independent groups | **{n_groups}** | the honest sample size — "
         "images inside a group are correlated |"),
        f"| Accuracy | {pct(m['accuracy'])} | at threshold {threshold:.2f} |",
        f"| Precision | {pct(m['precision'])} | of everything called fake |",
        f"| Recall | {pct(m['recall'])} | of the fakes present |",
        f"| F1 | {pct(m['f1'])} | |",
        f"| Specificity | {pct(m['specificity'])} | of the real images |",
        f"| ROC-AUC | {num(m['roc_auc'])} | ranking quality |",
        f"| PR-AUC | {num(m['pr_auc'])} | |",
        f"| **False-positive rate** | {pct(m['fpr'])} | **a real photograph called fake** |",
        f"| False-negative rate | {pct(m['fnr'])} | a deepfake called real |",
        f"| Brier score | {num(M.brier(y, s))} | calibration, 0 is perfect |",
        f"| ECE | {num(M.ece(y, s))} | calibration error |",
    ]

    if seen:
        seen_set = {x.strip().lower() for x in seen if x.strip()}
        in_domain = [r for r in rows if r["source"].lower() in seen_set]
        unseen = [r for r in rows if r["source"].lower() not in seen_set]
        if in_domain and unseen:
            a = M.evaluate(*split(in_domain), threshold)
            b = M.evaluate(*split(unseen), threshold)
            lines.append(f"| In-domain accuracy | {pct(a['accuracy'])} | "
                         f"{', '.join(sorted(seen_set))} |")
            lines.append(f"| **Cross-dataset accuracy** | {pct(b['accuracy'])} | "
                         f"generators never trained on |")
            gap = (a["accuracy"] - b["accuracy"]) * 100
            lines.append(f"| Generalisation gap | **{gap:+.2f} points** | "
                         "the number that matters most |")
        else:
            lines.append("| Cross-dataset accuracy | *no unseen source present* | "
                         "add a generator the model never trained on |")
    else:
        lines.append("| Cross-dataset accuracy | *not measured* | "
                     "needs `--seen` and a generator outside it |")

    if latency:
        for label, value in latency.items():
            lines.append(f"| {label} | **{value}** | `scripts/benchmark.py` |")

    lines += ["", "Per source:", "", "| Class | Source | n | Mean P(fake) | Outcome |",
              "|---|---|---|---|---|"]
    sources = {}
    for r in rows:
        sources.setdefault((r["label"], r["source"]), []).append(r)
    for (label, source), group in sorted(sources.items()):
        ys, ss = split(group)
        gm = M.evaluate(ys, ss, threshold)
        rate = gm["fpr"] if label == "real" else gm["recall"]
        outcome = (f"{rate * 100:.2f}% called fake" if label == "real"
                   else f"{rate * 100:.2f}% detected") if rate is not None else "—"
        lines.append(f"| {label} | `{source}` | {len(group)} | "
                     f"{sum(ss) / len(ss):.3f} | {outcome} |")

    return "\n".join(lines)


# -------------------------------------------------------------------- entry

def _resolved_cli_path(value):
    return Path(value).resolve() if value else None


def validate_v5_cli_args(parser, args):
    """Enforce the calibration -> validation -> sealed-test state machine."""
    stage_flags = args.fit_temperature or args.select_thresholds
    artifact_flags = any((args.calibration_in, args.calibration_out,
                          args.thresholds_in, args.thresholds_out))
    if (stage_flags or artifact_flags) and not args.manifest:
        parser.error("V5 calibration/threshold artifacts require --manifest")
    if not args.manifest:
        return
    if not args.from_csv:
        parser.error("--manifest requires --from-csv")
    if not args.dataset_root:
        parser.error("--manifest requires --dataset-root for path/hash validation")
    if not args.model_id and not args.model_artifact:
        parser.error("--manifest requires --model-id or --model-artifact")
    if args.fit_temperature and args.select_thresholds:
        parser.error("temperature fitting and threshold selection are separate commands")

    if args.fit_temperature:
        if args.split != "calibration":
            parser.error("--fit-temperature requires --split calibration")
        if not args.calibration_out:
            parser.error("--fit-temperature requires --calibration-out")
        if any((args.calibration_in, args.thresholds_in, args.thresholds_out,
                args.json_report, args.report, args.threshold is not None)):
            parser.error("temperature fitting may only write --calibration-out")
    elif args.select_thresholds:
        if args.split != "validation":
            parser.error("--select-thresholds requires --split validation")
        if not args.calibration_in or not args.thresholds_out:
            parser.error(
                "--select-thresholds requires --calibration-in and --thresholds-out")
        if any((args.calibration_out, args.thresholds_in, args.json_report,
                args.report, args.threshold is not None)):
            parser.error("threshold selection may only write --thresholds-out")
    else:
        if args.calibration_out:
            parser.error("--calibration-out is only valid with --fit-temperature")
        if args.thresholds_out:
            parser.error("--thresholds-out is only valid with --select-thresholds")
        evaluated_split = args.split or "sealed_test"
        if evaluated_split == "sealed_test" and (
                not args.calibration_in or not args.thresholds_in):
            parser.error(
                "sealed_test is evaluation-only and requires frozen "
                "--calibration-in and --thresholds-in artifacts")
        if evaluated_split == "sealed_test" and args.threshold is not None:
            parser.error("sealed_test threshold comes only from --thresholds-in")
        if args.thresholds_in and not args.calibration_in:
            parser.error("--thresholds-in requires the calibration artifact it was selected with")

    inputs = [args.from_csv, args.manifest, args.calibration_in,
              args.thresholds_in, args.model_artifact]
    outputs = [args.calibration_out, args.thresholds_out,
               args.json_report, args.report]
    input_paths = {_resolved_cli_path(value) for value in inputs if value}
    for output in outputs:
        if output and _resolved_cli_path(output) in input_paths:
            parser.error(f"output path must not overwrite an input or frozen artifact: {output}")

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", default=os.path.join(ROOT, "eval_data"),
                    help="folder holding real/<source> and fake/<source>")
    ap.add_argument("--from-csv", help="skip scoring, report on an existing CSV")
    ap.add_argument("--manifest", help="V5 manifest CSV; joins labels and split metadata to --from-csv")
    ap.add_argument("--dataset-root", help="dataset root used to validate V5 media paths and symlink confinement")
    ap.add_argument("--split", choices=("train", "calibration", "validation", "sealed_test", "all"),
                    help="V5 manifest split to report (default: sealed_test with --manifest)")
    ap.add_argument("--out", default=os.path.join(ROOT, "eval_data", "predictions.csv"),
                    help="where to write predictions")
    ap.add_argument("--threshold", type=float,
                    help="binary threshold for non-sealed reporting (default 0.5)")
    ap.add_argument("--target-fpr", type=float, default=0.01)
    ap.add_argument("--seen", help="comma-separated sources the model trained on")
    ap.add_argument("--limit", type=int, help="score only the first N images")
    ap.add_argument("--report", metavar="FILE",
                    help="write the benchmark table as markdown")
    ap.add_argument("--json-report", metavar="FILE",
                    help="write the complete V5 metrics/reliability/CI report as JSON")
    ap.add_argument("--bootstrap", type=int, default=200,
                    help="bootstrap resamples for V5 confidence intervals (0 disables; default 200)")
    ap.add_argument("--fit-temperature", action="store_true",
                    help="fit temperature scaling on the manifest calibration split only")
    ap.add_argument("--calibration-in", metavar="FILE",
                    help="apply a previously stored calibration record without refitting")
    ap.add_argument("--calibration-out", metavar="FILE",
                    help="where to write a newly fitted temperature-scaling record")
    ap.add_argument("--select-thresholds", action="store_true",
                    help="select real/fake/inconclusive thresholds on validation only")
    ap.add_argument("--thresholds-in", metavar="FILE",
                    help="load a frozen validation-only abstention-threshold artifact")
    ap.add_argument("--thresholds-out", metavar="FILE",
                    help="write thresholds selected on validation (selection command only)")
    model = ap.add_mutually_exclusive_group()
    model.add_argument("--model-id",
                       help="stable model/checkpoint identifier recorded in every artifact")
    model.add_argument("--model-artifact", metavar="FILE",
                       help="model/checkpoint file whose SHA-256 binds every artifact")
    ap.add_argument("--target-fnr", type=float, default=0.01,
                    help="validation FNR budget for the real threshold (default 0.01)")
    ap.add_argument("--minimum-abstention-band", type=float, default=0.10,
                    help="minimum score width reserved for inconclusive decisions (default 0.10)")
    ap.add_argument("--conditions", metavar="SRC_DIR",
                    help="generate processed variants instead of evaluating")
    args = ap.parse_args()

    if args.conditions:
        out = args.out
        if out.endswith(".csv"):                  # --out defaults to the CSV path
            out = os.path.join(ROOT, "eval_data", "conditions")
        return make_conditions(args.conditions, out)

    if args.bootstrap < 0:
        ap.error("--bootstrap must be non-negative")
    validate_v5_cli_args(ap, args)
    if args.threshold is None:
        args.threshold = M.DEFAULT_THRESHOLD

    print("DeepShield evaluation")

    if args.from_csv:
        rows = (read_manifest_predictions(args.from_csv, args.manifest, args.dataset_root)
                if args.manifest else read_csv(args.from_csv))
        print(f"  {len(rows)} predictions from {args.from_csv}")
    else:
        if not os.path.isdir(args.data):
            sys.exit(f"no evaluation data at {args.data}\n"
                     f"See eval_data/README.md for the expected layout.")
        items = collect(args.data)
        if not items:
            sys.exit(f"{args.data} has no images under real/ or fake/")
        import inference
        info = inference.engine_info()
        print(f"  model  {info.get('model_name')} {info.get('version')} "
              f"({info.get('runtime')})")
        print(f"  found  {len(items)} images")
        rows = score_all(items, load_group_overrides(args.data), args.out, args.limit)

    if args.manifest:
        context = artifact_context(
            args.manifest, model_id=args.model_id,
            model_artifact=args.model_artifact)
        if args.fit_temperature:
            fitted = fit_manifest_temperature(rows)
            artifact = calibration_artifact(fitted, context)
            write_artifact(args.calibration_out, artifact)
            print(f"  calibration artifact -> {args.calibration_out} (calibration split only)")
            return 0

        if args.select_thresholds:
            calibration_record = load_artifact(
                args.calibration_in, CALIBRATION_ARTIFACT, context)
            validation_rows = _rows_for_split(rows, "validation")
            apply_manifest_temperature(validation_rows, calibration_record)
            policy = select_manifest_policy(
                validation_rows, target_fpr=args.target_fpr,
                target_fnr=args.target_fnr,
                minimum_band=args.minimum_abstention_band)
            artifact = threshold_artifact(
                policy, context, _sha256_file(args.calibration_in))
            write_artifact(args.thresholds_out, artifact)
            print(f"  threshold artifact -> {args.thresholds_out} (validation split only)")
            return 0

        evaluated_split = args.split or "sealed_test"
        selected = _rows_for_split(rows, evaluated_split)
        calibration_record = None
        threshold_record = None
        decision_policy = None
        evaluation_threshold = args.threshold
        if args.calibration_in:
            calibration_record = load_artifact(
                args.calibration_in, CALIBRATION_ARTIFACT, context)
            apply_manifest_temperature(selected, calibration_record)
            print(f"  applied frozen calibration artifact from {args.calibration_in}")
        if args.thresholds_in:
            threshold_record = load_artifact(
                args.thresholds_in, THRESHOLD_ARTIFACT, context,
                calibration_sha256=_sha256_file(args.calibration_in))
            decision_policy = threshold_record["policy"]
            evaluation_threshold = float(decision_policy["fake_threshold"])
            print(f"  applied frozen threshold artifact from {args.thresholds_in}")
        v5 = v5_report(
            selected, evaluation_threshold, args.bootstrap, decision_policy,
            calibrated=calibration_record is not None)
        v5["evaluated_split"] = evaluated_split
        v5["artifact_context"] = context
        v5["calibration_artifact"] = calibration_record
        v5["threshold_artifact"] = threshold_record
        print_v5_summary(v5, evaluated_split)
        if args.json_report:
            with open(args.json_report, "w", encoding="utf-8", newline="\n") as stream:
                json.dump(v5, stream, indent=2, sort_keys=True)
                stream.write("\n")
            print(f"  V5 JSON report -> {args.json_report}")
        if args.report:
            table = markdown_report(selected, evaluation_threshold)
            with open(args.report, "w", encoding="utf-8", newline="\n") as f:
                f.write(table + "\n")
            print(f"\n  markdown table -> {args.report}")
        return 0

    seen = args.seen.split(",") if args.seen else None
    report(rows, args.threshold, seen=seen, target_fpr=args.target_fpr)

    if args.report:
        table = markdown_report(rows, args.threshold, seen=seen)
        with open(args.report, "w", encoding="utf-8", newline="\n") as f:
            f.write(table + "\n")
        print(f"\n  markdown table -> {args.report}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ManifestValidationError, TypeError, ValueError) as exc:
        print(f"V5 evaluation blocked: {exc}", file=sys.stderr)
        sys.exit(2)
