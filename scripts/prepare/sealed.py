"""The seal: data that must never reach training, and the guard that proves it.

Four things are evaluation-only:

    Celeb-DF v2              an entire dataset the model must never have seen
    DeeperForensics-1.0      the robustness ladder
    one FF++ manipulation    the cross-manipulation holdout
    the phone-photo test set the false-positive probe

If any of them leaks into training, every number produced afterwards is
worthless and — this is the dangerous part — nothing about the run will look
wrong. Training succeeds, accuracy climbs, and the result is meaningless.

So the guard is built to be **hard to defeat by accident**, which is how this
actually fails: a glob that catches one directory too many, a symlink, a
cached crop from an earlier run, a manifest reused after the seal changed.

Three independent layers, and any one of them failing stops the build:

    1. dataset name      Celeb-DF and DeeperForensics are sealed wholesale
    2. manipulation name the withheld FF++ method, wherever it appears
    3. group id          an explicit list, committed to the repository

Layer 3 is the important one. Names can be renamed and directories moved; the
committed group list is the record of what was sealed, written before
training and diffable afterwards.
"""
import os

from schema import Status

__all__ = ["SealedRegistry", "SealError", "DEFAULT_SEALED_DATASETS",
           "DEFAULT_WITHHELD_FFPP_METHOD"]

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Committed, deliberately outside the gitignored datasets/ tree so the seal
# cannot be lost by cleaning a working directory.
SEALED_GROUPS_FILE = os.path.join(ROOT, "docs", "v4_sealed_groups.txt")

DEFAULT_SEALED_DATASETS = ("celebdf", "deeperforensics")
DEFAULT_SEALED_SPLITS = ("phone_sealed",)

# Rotate this between runs to test a different cross-manipulation holdout.
# Whatever it is, it must be decided *before* training and never changed to
# make a number look better.
DEFAULT_WITHHELD_FFPP_METHOD = "neuraltextures"


class SealError(AssertionError):
    """Raised loudly. Never caught inside the pipeline."""


class SealedRegistry:
    """Decides what is sealed, and refuses to be talked out of it."""

    def __init__(self, sealed_datasets=DEFAULT_SEALED_DATASETS,
                 withheld_ffpp_method=DEFAULT_WITHHELD_FFPP_METHOD,
                 sealed_splits=DEFAULT_SEALED_SPLITS,
                 groups_file=SEALED_GROUPS_FILE):
        self.sealed_datasets = tuple(d.lower() for d in sealed_datasets)
        self.withheld = (withheld_ffpp_method or "").lower()
        self.sealed_splits = tuple(sealed_splits)
        self.groups_file = groups_file
        self.sealed_groups = self._load_groups()

    # ---- the committed group list

    def _load_groups(self):
        if not os.path.exists(self.groups_file):
            return set()
        with open(self.groups_file, encoding="utf-8") as f:
            return {line.strip() for line in f
                    if line.strip() and not line.startswith("#")}

    def record_groups(self, groups):
        """Write the seal. Additive — a group, once sealed, stays sealed.

        Un-sealing has to be a deliberate edit to a committed file that shows
        up in a diff, not a side effect of re-running the builder."""
        merged = self.sealed_groups | {str(g) for g in groups if g}
        os.makedirs(os.path.dirname(self.groups_file), exist_ok=True)
        with open(self.groups_file, "w", encoding="utf-8", newline="\n") as f:
            f.write("# DeepShield V4 — sealed evaluation groups.\n")
            f.write("# Written before training. Never split, never trained on.\n")
            f.write("# Removing a line here un-seals data: do it deliberately,\n")
            f.write("# in a commit, or not at all.\n")
            for group in sorted(merged):
                f.write(group + "\n")
        self.sealed_groups = merged
        return len(merged)

    # ---- the three layers

    def reason(self, row):
        """Why this row is sealed, or None. Works on ManifestRow or dict."""
        get = (lambda k: getattr(row, k, "")) if hasattr(row, "dataset") \
            else (lambda k: row.get(k, ""))

        dataset = str(get("dataset") or "").lower()
        if any(dataset.startswith(s) for s in self.sealed_datasets):
            return f"sealed dataset: {dataset}"

        method = str(get("manipulation_method") or "").lower()
        if self.withheld and method == self.withheld:
            return f"withheld FF++ method: {method}"

        group = str(get("group_id") or "")
        if group and group in self.sealed_groups:
            return f"sealed group: {group}"

        if str(get("split") or "") in self.sealed_splits:
            return f"sealed split: {get('split')}"

        return None

    def is_sealed(self, row):
        return self.reason(row) is not None

    # ---- the guard

    def assert_clean(self, rows, where="training manifest"):
        """Fail loudly if any sealed sample reached `rows`.

        Called on the training and validation manifests before either is
        written. There is no flag to skip it."""
        offenders = []
        for row in rows:
            reason = self.reason(row)
            if reason:
                get = (lambda k: getattr(row, k, "")) if hasattr(row, "dataset") \
                    else (lambda k: row.get(k, ""))
                offenders.append(f"{get('sample_id')} ({reason})")

        if offenders:
            shown = "\n  ".join(offenders[:20])
            more = f"\n  ... and {len(offenders) - 20} more" if len(offenders) > 20 else ""
            raise SealError(
                f"SEALED DATA IN {where.upper()} - {len(offenders)} sample(s).\n"
                f"  {shown}{more}\n"
                "Every number produced after this point would be worthless, and "
                "nothing about the training run would look wrong. Build stopped.")
        return True

    def partition(self, rows):
        """(trainable, sealed) — status is set on the sealed rows, not dropped."""
        trainable, sealed = [], []
        for row in rows:
            reason = self.reason(row)
            if reason:
                if hasattr(row, "status"):
                    row.status = Status.SEALED
                    row.notes = (row.notes + "; " if row.notes else "") + reason
                else:
                    row["status"] = Status.SEALED
                    row["notes"] = reason
                sealed.append(row)
            else:
                trainable.append(row)
        return trainable, sealed

    def describe(self):
        return {
            "sealed_datasets": list(self.sealed_datasets),
            "withheld_ffpp_method": self.withheld,
            "sealed_splits": list(self.sealed_splits),
            "sealed_groups": len(self.sealed_groups),
            "groups_file": os.path.relpath(self.groups_file, ROOT).replace(os.sep, "/"),
        }
