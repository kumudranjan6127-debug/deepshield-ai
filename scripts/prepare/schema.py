"""The manifest: what a row means, and what may never happen to one.

One rule shapes this file. **A manifest row is never deleted.** A sample that
turns out to be unreadable, duplicated, sealed or faceless keeps its row and
gains a status and a reason. A corpus that silently shrinks is one nobody can
audit later, and "we dropped 12,000 frames" is a sentence you want to be able
to answer.

The second rule: **the original label is never overwritten.** FF++ says
`Deepfakes`, DFDC says `FAKE`, Celeb-DF says `Celeb-synthesis`. Those go in
`original_label` and stay there. `normalized_label` is derived alongside it,
never in place of it, so a mapping mistake is recoverable rather than baked
in.
"""
from dataclasses import dataclass, field, asdict, fields
import csv
import os

__all__ = ["FIELDS", "Status", "Reject", "Label", "Family",
           "ManifestRow", "normalise_label", "write_manifest", "read_manifest"]


# --------------------------------------------------------------- vocabulary

class Status:
    """What happened to this sample. Every row has exactly one."""
    ACCEPTED = "ACCEPTED"          # usable, eligible for train/validation
    SEALED = "SEALED"              # evaluation only — must never train
    REJECTED = "REJECTED"          # see rejection_reason
    PENDING = "PENDING"            # inventoried, not yet extracted

    ALL = (ACCEPTED, SEALED, REJECTED, PENDING)


class Reject:
    """Why a sample is not usable. Recorded, never silently applied."""
    NO_FACE = "NO_FACE"
    FACE_TOO_SMALL = "FACE_TOO_SMALL"
    MULTIPLE_FACE_POLICY = "MULTIPLE_FACE_POLICY"
    LOW_DETECTION_SCORE = "LOW_DETECTION_SCORE"
    CORRUPT_FILE = "CORRUPT_FILE"
    UNREADABLE_VIDEO = "UNREADABLE_VIDEO"
    INVALID_LABEL = "INVALID_LABEL"
    INVALID_DIMENSIONS = "INVALID_DIMENSIONS"
    INVALID_DURATION = "INVALID_DURATION"
    EXACT_DUPLICATE = "EXACT_DUPLICATE"
    NEAR_DUPLICATE = "NEAR_DUPLICATE"
    UNSAFE_GROUP = "UNSAFE_GROUP"   # provenance too weak to split on — see below

    ALL = (NO_FACE, FACE_TOO_SMALL, MULTIPLE_FACE_POLICY, LOW_DETECTION_SCORE,
           CORRUPT_FILE, UNREADABLE_VIDEO, INVALID_LABEL, INVALID_DIMENSIONS,
           INVALID_DURATION, EXACT_DUPLICATE, NEAR_DUPLICATE, UNSAFE_GROUP)


class Label:
    REAL = "REAL"
    FAKE = "FAKE"
    UNKNOWN = "UNKNOWN"
    ALL = (REAL, FAKE, UNKNOWN)


class Family:
    """How the fake was made. `UNKNOWN_FAKE` is a legitimate answer.

    Guessing a family from a filename would put a wrong fact in a manifest
    that later becomes a table in the model card."""
    REAL = "REAL"
    FACE_SWAP = "FACE_SWAP"
    FACE_REENACTMENT = "FACE_REENACTMENT"
    GAN = "GAN"
    DIFFUSION = "DIFFUSION"
    UNKNOWN_FAKE = "UNKNOWN_FAKE"
    ALL = (REAL, FACE_SWAP, FACE_REENACTMENT, GAN, DIFFUSION, UNKNOWN_FAKE)


# Only mappings established from each dataset's own documentation. Anything
# absent from this table resolves to UNKNOWN_FAKE rather than a guess.
MANIPULATION_FAMILY = {
    # FaceForensics++ — the four methods, per the paper
    "deepfakes": Family.FACE_SWAP,
    "faceswap": Family.FACE_SWAP,
    "faceshifter": Family.FACE_SWAP,
    "face2face": Family.FACE_REENACTMENT,
    "neuraltextures": Family.FACE_REENACTMENT,
    # DFDC — the release does not name per-clip methods
    "dfdc": Family.FACE_SWAP,
    # Celeb-DF
    "celeb-synthesis": Family.FACE_SWAP,
    # Fully generated
    "stylegan": Family.GAN,
    "stylegan2": Family.GAN,
    "tpdn": Family.GAN,
    "stable-diffusion": Family.DIFFUSION,
    "diffusion": Family.DIFFUSION,
    # Real
    "none": Family.REAL,
    "": Family.REAL,
}

# Dataset label strings → REAL/FAKE. Case-insensitive.
LABEL_ALIASES = {
    "real": Label.REAL, "original": Label.REAL, "pristine": Label.REAL,
    "youtube": Label.REAL, "actors": Label.REAL, "celeb-real": Label.REAL,
    "fake": Label.FAKE, "manipulated": Label.FAKE, "synthesis": Label.FAKE,
    "celeb-synthesis": Label.FAKE, "deepfakes": Label.FAKE,
    "face2face": Label.FAKE, "faceswap": Label.FAKE,
    "neuraltextures": Label.FAKE, "faceshifter": Label.FAKE,
}


def normalise_label(original_label, manipulation=""):
    """(normalized_label, manipulation_family) — the original is untouched.

    Returns UNKNOWN / UNKNOWN_FAKE rather than guessing. An unmapped label is
    a gap in the mapping table, and it should look like one."""
    key = str(original_label or "").strip().lower()
    label = LABEL_ALIASES.get(key, Label.UNKNOWN)

    method = str(manipulation or "").strip().lower()
    if label == Label.REAL:
        return label, Family.REAL
    if label == Label.FAKE:
        return label, MANIPULATION_FAMILY.get(method or key, Family.UNKNOWN_FAKE)
    return Label.UNKNOWN, Family.UNKNOWN_FAKE


# ------------------------------------------------------------------- a row

@dataclass
class ManifestRow:
    sample_id: str                      # stable, derived from dataset+path+frame
    dataset: str
    source_path: str
    source_id: str = ""                 # file stem, dataset-native
    video_id: str = ""                  # empty for stills
    frame_index: int = -1               # -1 = not a video frame
    timestamp: float = -1.0
    subject_id: str = ""                # when the dataset provides one
    group_id: str = ""                  # THE split key. Never guessed.
    group_source: str = ""              # how group_id was established
    original_label: str = ""            # verbatim from the dataset
    normalized_label: str = Label.UNKNOWN
    manipulation_method: str = ""
    manipulation_family: str = Family.UNKNOWN_FAKE
    compression: str = ""               # raw / c23 / c40 / jpeg / unknown
    media_type: str = "image"           # image | video
    width: int = -1
    height: int = -1
    duration: float = -1.0
    frame_count: int = -1
    sha256: str = ""
    phash: str = ""
    face_found: bool = False
    face_score: float = -1.0
    face_count: int = 0
    crop_box: str = ""                  # "x,y,w,h" in capped-frame coords
    crop_path: str = ""
    split: str = ""                     # train | validation | sealed | ""
    status: str = Status.PENDING
    rejection_reason: str = ""
    notes: str = ""

    def as_dict(self):
        d = asdict(self)
        d["face_found"] = "1" if self.face_found else "0"
        return d


FIELDS = [f.name for f in fields(ManifestRow)]


# ------------------------------------------------------------------- io

def write_manifest(path, rows):
    """Rows are appended to the record, never pruned from it."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.as_dict() if isinstance(row, ManifestRow) else row)
    return path


def read_manifest(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["face_found"] = r.get("face_found") in ("1", "True", "true")
        for key, cast in (("frame_index", int), ("face_count", int),
                          ("width", int), ("height", int), ("frame_count", int),
                          ("timestamp", float), ("duration", float),
                          ("face_score", float)):
            try:
                r[key] = cast(r.get(key, -1))
            except (TypeError, ValueError):
                r[key] = -1 if cast is int else -1.0
    return rows
