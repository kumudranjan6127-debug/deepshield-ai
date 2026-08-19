"""Shared, evaluation-only dataset manifest utilities.

Nothing in this module downloads, crops, trains on, or otherwise changes
media.  A label is accepted only from explicit metadata supplied by the data
custodian; the directory and filename are never used as prediction signals or
as a source of ground truth.
"""
from __future__ import annotations

import csv
import hashlib
from collections import Counter, defaultdict
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
MANIFEST_FIELDS = [
    "relative_path", "label", "media_type", "source_dataset", "source_id",
    "identity_id", "manipulation_type", "provenance", "usage_note", "sha256",
    "width", "height", "status", "validation_errors",
]
METADATA_FIELDS = MANIFEST_FIELDS[:9]


def normalized_path(value: str) -> str:
    return Path(str(value or "").replace("\\", "/")).as_posix().lstrip("./")


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
        key = normalized_path(row.get("relative_path", ""))
        if not key:
            errors.append(f"metadata row {number}: missing relative_path")
        elif key in index:
            errors.append(f"metadata row {number}: duplicate relative_path {key}")
        else:
            index[key] = {field: str(row.get(field, "") or "").strip()
                          for field in METADATA_FIELDS}
    return index, errors


def image_details(path: Path) -> tuple[int | str, int | str, str]:
    """Validate an image before returning dimensions; no decoder guesswork."""
    try:
        from PIL import Image
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            width, height = image.size
        return width, height, ""
    except Exception as exc:
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
