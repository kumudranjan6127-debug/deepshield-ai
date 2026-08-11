"""Walk the raw datasets, sample frames, extract faces, record everything.

Three jobs, in order, all of them recording rather than discarding:

    inventory  every source file becomes a manifest row
    sample     videos become frames, deterministically
    extract    frames become face crops, through the production path

Face detection is **not reimplemented here.** It calls
`training/deepshield_preprocess.py`, which is itself pinned bit-identical to
production by `tests/test_preprocess_parity.py`. A third copy of the crop
logic would be a third thing to keep in sync.
"""
import hashlib
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "training"))

from schema import (Family, Label, ManifestRow, Reject,  # noqa: E402
                    Status, normalise_label)

IMAGE_EXT = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
VIDEO_EXT = (".mp4", ".mov", ".avi", ".webm", ".mkv")

MIN_FACE_PX = 64            # short side of the detected box
MIN_DETECTION_SCORE = 0.6   # matches production's YuNet threshold


def relative_to_root(path):
    """Repo-relative when possible, absolute otherwise.

    `os.path.relpath` raises across Windows drive letters, and a dataset on
    a second disk is an entirely reasonable thing to have. Manifests stay
    portable when they can and stay correct when they cannot."""
    try:
        return os.path.relpath(path, ROOT).replace(os.sep, "/")
    except ValueError:
        return os.path.abspath(path).replace(os.sep, "/")


# ------------------------------------------------------------------ hashing

def sha256_of(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def phash_of(pil_image, size=8):
    """64-bit difference hash — cheap, and enough to spot near-identical
    frames of one shot. Not a similarity metric between different people."""
    import numpy as np
    from PIL import Image
    grey = pil_image.convert("L").resize((size + 1, size), Image.LANCZOS)
    arr = np.asarray(grey, dtype=np.int16)
    bits = arr[:, 1:] > arr[:, :-1]
    value = 0
    for bit in bits.flatten():
        value = (value << 1) | int(bit)
    return f"{value:016x}"


def hamming(a, b):
    if not a or not b or len(a) != len(b):
        return 64
    return bin(int(a, 16) ^ int(b, 16)).count("1")


# ---------------------------------------------------------------- inventory

def stable_id(dataset, source_path, frame_index=-1):
    """Deterministic and collision-resistant: the same file gives the same id
    on every machine and every run, which is what makes the manifest
    diffable."""
    key = f"{dataset}|{source_path}|{frame_index}"
    return f"{dataset}_{hashlib.sha1(key.encode()).hexdigest()[:16]}"


def inventory_directory(root, dataset, label_of, group_of, method_of=None,
                        compression_of=None, subject_of=None, limit=None):
    """Every file under `root` becomes a row. Nothing is skipped silently.

    The four callables are what makes this dataset-agnostic: each dataset
    supplies its own rules for label, group, method and compression, and this
    function knows none of them."""
    rows = []
    if not os.path.isdir(root):
        return rows

    for dirpath, _, filenames in os.walk(root):
        for name in sorted(filenames):
            path = os.path.join(dirpath, name)
            ext = os.path.splitext(name)[1].lower()
            if ext not in IMAGE_EXT + VIDEO_EXT:
                continue

            rel = relative_to_root(path)
            original = label_of(path) or ""
            method = (method_of(path) if method_of else "") or ""
            normalized, family = normalise_label(original, method)

            group = group_of(path)
            row = ManifestRow(
                sample_id=stable_id(dataset, rel),
                dataset=dataset,
                source_path=rel,
                source_id=os.path.splitext(name)[0],
                media_type="video" if ext in VIDEO_EXT else "image",
                original_label=original,
                normalized_label=normalized,
                manipulation_method=method,
                manipulation_family=family,
                compression=(compression_of(path) if compression_of else "") or "unknown",
                subject_id=(subject_of(path) if subject_of else "") or "",
                group_id=group or "",
                group_source="dataset metadata" if group else "NONE",
                status=Status.PENDING,
            )

            # Provenance too weak to split on is a rejection, not a guess.
            # Guessing here is how an identity ends up on both sides.
            if not group:
                row.status = Status.REJECTED
                row.rejection_reason = Reject.UNSAFE_GROUP
                row.notes = "no group key could be established from metadata"
            if normalized == Label.UNKNOWN:
                row.status = Status.REJECTED
                row.rejection_reason = Reject.INVALID_LABEL
                row.notes = f"unmapped label {original!r}"

            rows.append(row)
            if limit and len(rows) >= limit:
                return rows
    return rows


# ------------------------------------------------------------ frame sampling

def sample_frame_indices(frame_count, fps, target_frames, min_gap_seconds=0.0):
    """Uniform temporal sampling — deterministic, never random.

    The beginning of a clip is not representative of it, so this spreads
    across the whole duration. A fixed rule means two runs on the same file
    choose the same frames, which is what makes an extraction reproducible.
    """
    if frame_count <= 0 or target_frames <= 0:
        return []
    if frame_count <= target_frames:
        return list(range(frame_count))

    step = frame_count / float(target_frames)
    if fps and min_gap_seconds:
        step = max(step, fps * min_gap_seconds)

    indices, position = [], step / 2.0     # centre of the first bucket
    while position < frame_count and len(indices) < target_frames:
        indices.append(int(position))
        position += step
    return sorted(set(indices))


def probe_video(path):
    """(frame_count, fps, width, height, duration) or None if unreadable."""
    import cv2
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps != fps or fps <= 0:
            fps = 25.0
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        return count, float(fps), width, height, (count / fps if fps else 0.0)
    finally:
        cap.release()


def expand_video_rows(row, target_frames, min_gap_seconds=0.0):
    """One video row → one row per sampled frame.

    The video's own row stays in the manifest as the parent record; the frame
    rows inherit its label, method and — critically — its group."""
    path = os.path.join(ROOT, row.source_path)
    probed = probe_video(path)
    if not probed:
        row.status = Status.REJECTED
        row.rejection_reason = Reject.UNREADABLE_VIDEO
        return []

    count, fps, width, height, duration = probed
    row.frame_count, row.width, row.height, row.duration = count, width, height, duration
    if count <= 0:
        row.status = Status.REJECTED
        row.rejection_reason = Reject.INVALID_DURATION
        return []

    frames = []
    for index in sample_frame_indices(count, fps, target_frames, min_gap_seconds):
        child = ManifestRow(**{**row.__dict__})
        child.sample_id = stable_id(row.dataset, row.source_path, index)
        child.media_type = "image"
        child.video_id = row.source_id
        child.frame_index = index
        child.timestamp = round(index / fps, 3)
        child.frame_count = count
        child.duration = duration
        child.status = Status.PENDING
        child.rejection_reason = ""
        frames.append(child)
    return frames


def read_video_frame(path, index):
    """Grab past the frames we do not want; decode only the one we do."""
    import cv2
    from PIL import Image
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None
    try:
        for _ in range(index):
            if not cap.grab():
                return None
        ok, frame = cap.read()
        if not ok:
            return None
        return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()


# ------------------------------------------------------------- face extraction

class FaceExtractor:
    """Crops through the production path. Records why anything is rejected.

    Crops are written losslessly. A JPEG intermediate would bake one
    compression generation into every sample, which is exactly the shortcut
    V1 learned instead of learning generator artefacts."""

    def __init__(self, yunet_path, crops_dir, min_face_px=MIN_FACE_PX,
                 min_score=MIN_DETECTION_SCORE, multiple_face_policy="largest"):
        from deepshield_preprocess import Preprocessor
        self.pre = Preprocessor(yunet_path)
        self.crops_dir = crops_dir
        self.min_face_px = min_face_px
        self.min_score = min_score
        self.multiple_face_policy = multiple_face_policy

    def _load(self, row):
        from PIL import Image
        path = row.source_path if os.path.isabs(row.source_path)             else os.path.join(ROOT, row.source_path)
        if row.frame_index >= 0:
            return read_video_frame(path, row.frame_index)
        try:
            with Image.open(path) as im:
                return im.convert("RGB")
        except Exception:
            return None

    def extract(self, row):
        """Fill the detection fields, write the crop, set the status.

        Never returns None and never drops the row — a sample that fails
        here becomes a REJECTED row with a reason, so the corpus stays
        reconstructible."""
        image = self._load(row)
        if image is None:
            row.status = Status.REJECTED
            row.rejection_reason = (Reject.UNREADABLE_VIDEO if row.frame_index >= 0
                                    else Reject.CORRUPT_FILE)
            return row

        row.width, row.height = image.size
        if min(image.size) < 32:
            row.status = Status.REJECTED
            row.rejection_reason = Reject.INVALID_DIMENSIONS
            return row

        found = self.pre.detect_face(image)
        row.face_found = bool(found.found)
        row.face_count = int(found.n_faces)
        row.face_score = float(found.score) if found.score is not None else -1.0
        if found.box:
            row.crop_box = ",".join(f"{v:.1f}" for v in found.box)

        if not found.found:
            row.status = Status.REJECTED
            row.rejection_reason = Reject.NO_FACE
            row.notes = "kept for inspection; training on faceless frames " \
                        "teaches the model to classify backgrounds"
            return row

        if min(found.box[2], found.box[3]) < self.min_face_px:
            row.status = Status.REJECTED
            row.rejection_reason = Reject.FACE_TOO_SMALL
            return row

        if 0 <= row.face_score < self.min_score:
            row.status = Status.REJECTED
            row.rejection_reason = Reject.LOW_DETECTION_SCORE
            return row

        if self.multiple_face_policy == "reject" and found.n_faces > 1:
            row.status = Status.REJECTED
            row.rejection_reason = Reject.MULTIPLE_FACE_POLICY
            return row

        target = os.path.join(self.crops_dir, row.dataset,
                              _safe(row.group_id), row.sample_id + ".png")
        os.makedirs(os.path.dirname(target), exist_ok=True)
        found.crop.save(target, "PNG")            # lossless, deliberately

        row.crop_path = relative_to_root(target)
        row.phash = phash_of(found.crop)
        row.status = Status.ACCEPTED
        return row


def _safe(name):
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in str(name))[:80] or "ungrouped"
