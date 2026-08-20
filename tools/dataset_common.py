"""Shared, evaluation-only dataset manifest utilities.

Nothing in this module downloads, crops, trains on, or otherwise changes
media.  A label is accepted only from explicit metadata supplied by the data
custodian; the directory and filename are never used as prediction signals or
as a source of ground truth.
"""
from __future__ import annotations

import csv
import hashlib
import re
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath, PureWindowsPath

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
MEDIA_EXTENSIONS = IMAGE_EXTENSIONS | VIDEO_EXTENSIONS
SPLITS = ("train", "calibration", "validation", "sealed_test")
ROBUSTNESS_SLICES = ("clean", "jpeg", "resize", "blur", "screenshot", "low_light")
BOOLEAN_VALUES = {"0", "1", "false", "true", "no", "yes"}

# ``relative_path`` remains the public column name for compatibility with the
# existing benchmark and failure-analysis tools. It is the media path, always
# relative to the dataset root; accepting absolute paths would let a manifest
# turn a benchmark into an arbitrary-file reader.
MANIFEST_FIELDS = [
    "relative_path", "label", "media_type", "modality", "source_dataset",
    "source_id", "identity_id", "identity_group", "manipulation_type",
    "generator_family", "compression_slice", "robustness_slice", "split",
    "generator_disjoint", "provenance", "usage_note", "sha256", "width",
    "height", "status", "validation_errors",
]
# These are facts supplied by the custodian. File type, hashes, dimensions and
# validation state are observed locally and never inferred from a filename.
METADATA_FIELDS = [
    "relative_path", "label", "modality", "source_dataset", "source_id",
    "identity_id", "identity_group", "manipulation_type", "generator_family",
    "compression_slice", "robustness_slice", "split", "generator_disjoint",
    "provenance", "usage_note",
]


class ManifestValidationError(ValueError):
    """A manifest cannot safely support an evaluation claim."""


def normalized_path(value: str) -> str:
    """Return a portable spelling without hiding unsafe path components."""
    return str(value or "").strip().replace("\\", "/")


def relative_path_error(value: str) -> str:
    """Explain why a manifest/prediction path is not portable and relative."""
    original = str(value or "").strip()
    if not original:
        return "missing media path"
    if "\\" in original:
        return f"unsafe relative_path: use portable forward slashes: {value!r}"
    windows = PureWindowsPath(original)
    posix = PurePosixPath(original)
    if (windows.is_absolute() or windows.drive or posix.is_absolute()
            or original.startswith("//")):
        return f"unsafe relative_path: absolute paths are forbidden: {value!r}"
    if any(part in ("", ".", "..") for part in posix.parts):
        return f"unsafe relative_path: traversal/redundant components: {value!r}"
    if posix.as_posix() != original:
        return f"unsafe relative_path: path is not normalized: {value!r}"
    return ""


def safe_media_path(dataset: Path, value: str) -> tuple[Path | None, str]:
    """Resolve one manifest media path without allowing path or symlink escape.

    A valid manifest path is relative, has no ``..`` segment, resolves under
    ``dataset``, and traverses no symlink. Rejecting even an internal symlink
    keeps the file identity stable between manifest creation and scoring.
    """
    raw = str(value or "").strip()
    path_error = relative_path_error(raw)
    if path_error:
        return None, path_error
    candidate_rel = Path(raw)

    root = dataset.resolve()
    candidate = dataset.joinpath(candidate_rel)
    current = dataset
    for part in candidate_rel.parts:
        current = current / part
        if current.is_symlink():
            return None, f"symlink is not permitted in media path: {value!r}"
    try:
        resolved = candidate.resolve()
        resolved.relative_to(root)
    except (OSError, RuntimeError, ValueError):
        return None, f"media path escapes dataset root: {value!r}"
    return resolved, ""


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def metadata_by_path(path: Path) -> tuple[dict[str, dict[str, str]], list[str]]:
    """Return metadata keyed by relative path, without inventing any values."""
    if not path.exists():
        return {}, [f"metadata file not found: {path}"]
    rows = read_csv(path)
    missing = set(METADATA_FIELDS) - set(rows[0] if rows else [])
    errors = [f"metadata missing column: {field}" for field in sorted(missing)]
    index: dict[str, dict[str, str]] = {}
    for number, row in enumerate(rows, start=2):
        raw_path = row.get("relative_path", "")
        key = normalized_path(raw_path)
        path_error = relative_path_error(raw_path)
        if path_error:
            errors.append(f"metadata row {number}: {path_error}")
            continue
        if not key:
            errors.append(f"metadata row {number}: missing relative_path")
        elif key in index:
            errors.append(f"metadata row {number}: duplicate relative_path {key}")
        else:
            index[key] = {field: str(row.get(field, "") or "").strip()
                          for field in METADATA_FIELDS}
    return index, errors


def _truth(value: str) -> bool | None:
    value = str(value or "").strip().lower()
    if value in {"1", "true", "yes"}:
        return True
    if value in {"0", "false", "no"}:
        return False
    return None


def manifest_validation_errors(rows: list[dict], dataset: Path | None = None,
                               *, verify_hashes: bool = True) -> list[str]:
    """Return all split and path violations; never silently repair a row.

    The V5 contract intentionally checks every split pair. In particular,
    calibration cannot share a file or identity/original-video group with
    validation or the sealed test, and a generator-disjoint sealed test
    cannot contain a generator family that appeared in any other split.
    """
    if not rows:
        return ["manifest has no records"]
    errors: list[str] = [
        f"manifest missing required column: {field}"
        for field in sorted(set(MANIFEST_FIELDS) - set(rows[0]))
    ]
    paths: dict[str, list[tuple[int, str]]] = defaultdict(list)
    hashes: dict[str, list[tuple[int, str]]] = defaultdict(list)
    groups: dict[str, list[tuple[int, str]]] = defaultdict(list)
    sealed_families: dict[str, list[tuple[int, str]]] = defaultdict(list)
    nonsealed_families: dict[str, list[tuple[int, str]]] = defaultdict(list)

    for number, row in enumerate(rows, start=2):
        def value(field: str, source=row) -> str:
            return str(source.get(field, "") or "").strip()

        path = value("relative_path")
        split = value("split").lower()
        label = value("label").lower()
        media_type = value("media_type").lower()
        modality = value("modality").lower()
        group = value("identity_group") or value("identity_id")
        family = value("generator_family").lower()
        disjoint = _truth(value("generator_disjoint"))

        if not path:
            errors.append(f"row {number}: missing relative_path (media path)")
        else:
            normal = normalized_path(path)
            path_error = relative_path_error(path)
            if path_error:
                errors.append(f"row {number}: {path_error}")
            paths[normal].append((number, split))
            if dataset is not None:
                resolved, reason = safe_media_path(dataset, path)
                if reason:
                    errors.append(f"row {number}: {reason}")
                elif not resolved or not resolved.is_file():
                    errors.append(f"row {number}: media file does not exist: {path!r}")

        if label not in {"real", "fake"}:
            errors.append(f"row {number}: label must be 'real' or 'fake', got {label!r}")
        if split not in SPLITS:
            errors.append(f"row {number}: split must be one of {', '.join(SPLITS)}, got {split!r}")
        if modality not in {"image", "video"}:
            errors.append(f"row {number}: modality must be 'image' or 'video', got {modality!r}")
        if media_type not in {"image", "video"}:
            errors.append(
                f"row {number}: media_type must be 'image' or 'video', got {media_type!r}")
        elif modality and media_type != modality:
            errors.append(
                f"row {number}: media_type {media_type!r} does not match modality {modality!r}")
        for field in ("source_dataset", "identity_group", "manipulation_type",
                      "generator_family", "compression_slice", "provenance",
                      "robustness_slice"):
            if field == "identity_group" and value("identity_id"):
                continue
            if not value(field):
                errors.append(f"row {number}: missing required metadata: {field}")
        if value("robustness_slice").lower() not in ROBUSTNESS_SLICES:
            errors.append(
                f"row {number}: robustness_slice must be one of {', '.join(ROBUSTNESS_SLICES)}, "
                f"got {value('robustness_slice')!r}")
        if disjoint is None:
            errors.append(f"row {number}: generator_disjoint must be yes/no or true/false")
        if label == "fake":
            for field in ("manipulation_type", "generator_family"):
                if not value(field):
                    errors.append(f"row {number}: fake sample missing required metadata: {field}")
        elif family and family not in {"real", "none", "n/a"}:
            errors.append(f"row {number}: real sample generator_family must be real/none, got {family!r}")

        if group and split:
            groups[group].append((number, split))
        digest = value("sha256").lower()
        if not digest:
            errors.append(f"row {number}: missing required integrity metadata: sha256")
        elif not re.fullmatch(r"[0-9a-f]{64}", digest):
            errors.append(f"row {number}: sha256 must be 64 lowercase hexadecimal characters")
        observed_digest = digest
        if dataset is not None and verify_hashes and path and not relative_path_error(path):
            resolved, reason = safe_media_path(dataset, path)
            if not reason and resolved and resolved.is_file():
                actual_digest = sha256_of(resolved)
                observed_digest = actual_digest
                if digest and digest != actual_digest:
                    errors.append(
                        f"row {number}: manifest sha256 does not match media file {path!r}")
        if observed_digest and split:
            hashes[observed_digest].append((number, split))
        if label == "fake" and family:
            if split == "sealed_test" and disjoint:
                sealed_families[family].append((number, split))
            elif split != "sealed_test":
                nonsealed_families[family].append((number, split))

    for path, seen in sorted(paths.items()):
        if len({split for _, split in seen}) > 1:
            locations = ", ".join(f"row {number} ({split})" for number, split in seen)
            errors.append(f"file overlap across splits for {path!r}: {locations}")
    for digest, seen in sorted(hashes.items()):
        if len({split for _, split in seen}) > 1:
            locations = ", ".join(f"row {number} ({split})" for number, split in seen)
            errors.append(f"content-hash overlap across splits ({digest[:12]}...): {locations}")
    for group, seen in sorted(groups.items()):
        if len({split for _, split in seen}) > 1:
            locations = ", ".join(f"row {number} ({split})" for number, split in seen)
            errors.append(f"identity/original-video group overlap for {group!r}: {locations}")
    for family, sealed_rows in sorted(sealed_families.items()):
        if family in nonsealed_families:
            locations = ", ".join(
                f"row {number} ({split})"
                for number, split in sealed_rows + nonsealed_families[family])
            errors.append(
                f"generator-family leakage for generator-disjoint sealed test {family!r}: {locations}")
    return errors


def validate_manifest(rows: list[dict], dataset: Path | None = None) -> None:
    errors = manifest_validation_errors(rows, dataset)
    if errors:
        preview = "\n  - ".join(errors[:20])
        more = f"\n  - ... and {len(errors) - 20} more" if len(errors) > 20 else ""
        raise ManifestValidationError(f"V5 manifest validation failed:\n  - {preview}{more}")


def image_details(path: Path) -> tuple[int | str, int | str, str]:
    """Validate an image before returning dimensions; no decoder guesswork."""
    try:
        from PIL import Image
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            width, height = image.size
        return width, height, ""
    except Exception as exc:  # noqa: BLE001 - validation records decoder failures
        return "", "", f"invalid image: {type(exc).__name__}"


def dhash_of(path: Path) -> str:
    """A deterministic perceptual hash for reporting visually duplicate images."""
    import numpy as np
    from PIL import Image
    with Image.open(path) as image:
        grey = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
        pixels = np.asarray(grey, dtype=np.int16)
    bits = pixels[:, 1:] > pixels[:, :-1]
    value = 0
    for bit in bits.ravel():
        value = (value << 1) | int(bit)
    return f"{value:016x}"


def duplicate_groups(values: dict[str, str]) -> list[list[str]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for name, value in values.items():
        if value:
            grouped[value].append(name)
    return [sorted(group) for group in grouped.values() if len(group) > 1]


def distribution(rows: list[dict], field: str) -> dict[str, int]:
    counts = Counter(str(row.get(field, "") or "unknown") for row in rows)
    return dict(sorted(counts.items()))


def resolution_distribution(rows: list[dict]) -> dict[str, int]:
    buckets = Counter()
    for row in rows:
        try:
            side = max(int(row.get("width") or 0), int(row.get("height") or 0))
        except (TypeError, ValueError):
            side = 0
        if side <= 0:
            bucket = "unknown"
        elif side < 128:
            bucket = "under_128px"
        elif side < 256:
            bucket = "128_to_255px"
        elif side < 512:
            bucket = "256_to_511px"
        elif side < 1024:
            bucket = "512_to_1023px"
        else:
            bucket = "1024px_or_larger"
        buckets[bucket] += 1
    return dict(buckets)
