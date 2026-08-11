"""Duplicates and splits — both operating on groups, never on files.

The failure this file exists to prevent: a face that appears in training and
again in validation, so the reported number describes memorisation. It has
already happened once in this project. On synthetic data with DFDC's
structure, the previous file-level split put **78 of 100** held-out files in
the same identity as a training file.

Two defences, and the second is the real one:

    duplicates   the same image twice, possibly under different names
    groups       the same *person* twice, under entirely different images

Deduplication is reported, not enforced by deletion. A duplicate keeps its
manifest row and gains a status; what it must never do is cross a split
boundary.
"""
import os
import random
from collections import Counter, defaultdict

from inventory import hamming
from schema import Reject, Status

__all__ = ["find_duplicates", "assign_duplicate_status", "group_split",
           "assert_no_group_leakage", "composition"]

NEAR_DUPLICATE_DISTANCE = 4      # Hamming, on the 64-bit dHash


# ------------------------------------------------------------- duplicates

def find_duplicates(rows):
    """→ {"exact": [[sample_id, ...]], "near": [[...]], "cross_group": [...]}

    Near-duplicate search runs **inside a group only**. Two different people
    can hash close together, and dropping those loses real diversity; the
    group key is what stops contamination across splits. What this looks for
    inside a group is redundancy — 300 near-identical frames of one static
    shot are not 300 samples.

    Exact duplicates are checked globally, because the same file appearing in
    two datasets is a provenance fact worth surfacing."""
    usable = [r for r in rows if _get(r, "status") in (Status.ACCEPTED, Status.SEALED)]

    by_sha = defaultdict(list)
    for row in usable:
        digest = _get(row, "sha256")
        if digest:
            by_sha[digest].append(_get(row, "sample_id"))
    exact = [ids for ids in by_sha.values() if len(ids) > 1]

    # A duplicate spanning two groups is the dangerous kind: it means the
    # group key is wrong, and no split can be trusted until it is fixed.
    group_of = {_get(r, "sample_id"): _get(r, "group_id") for r in usable}
    cross_group = [ids for ids in exact
                   if len({group_of.get(i) for i in ids}) > 1]

    by_group = defaultdict(list)
    for row in usable:
        by_group[_get(row, "group_id")].append(row)

    near = []
    for group_rows in by_group.values():
        kept = []
        for row in sorted(group_rows, key=lambda r: _get(r, "sample_id")):
            digest = _get(row, "phash")
            if not digest:
                continue
            match = next((k for k in kept
                          if hamming(digest, _get(k, "phash")) <= NEAR_DUPLICATE_DISTANCE),
                         None)
            if match is None:
                kept.append(row)
            else:
                near.append([_get(match, "sample_id"), _get(row, "sample_id")])

    return {"exact": exact, "near": near, "cross_group": cross_group}


def assign_duplicate_status(rows, duplicates):
    """Mark the redundant copies. **Rows are kept**, statuses change.

    The first sample of each duplicate set survives; the rest become
    REJECTED with a reason, so the corpus is reconstructible and the count
    is auditable."""
    redundant = {}

    # A duplicate spanning two groups is a different problem from a duplicate
    # inside one: the same file carries two identities, so NEITHER assignment
    # can be trusted, and putting it on one side would silently place that
    # face on both. Every copy is rejected, not just the extras.
    #
    # Not hypothetical. The first run of this pipeline found one photograph
    # present in LFW under two different people's names.
    for ids in duplicates.get("cross_group", []):
        for sample_id in ids:
            redundant[sample_id] = Reject.UNSAFE_GROUP

    for ids in duplicates["exact"]:
        for sample_id in ids[1:]:
            redundant.setdefault(sample_id, Reject.EXACT_DUPLICATE)
    for keeper, dropped in duplicates["near"]:
        redundant.setdefault(dropped, Reject.NEAR_DUPLICATE)

    marked = 0
    for row in rows:
        reason = redundant.get(_get(row, "sample_id"))
        if reason and _get(row, "status") == Status.ACCEPTED:
            _set(row, "status", Status.REJECTED)
            _set(row, "rejection_reason", reason)
            marked += 1
    return marked


# ------------------------------------------------------------------ splits

def group_split(rows, validation_fraction=0.2, seed=42):
    """Deterministic split over **groups**, then rows follow their group.

    Groups are shuffled with a fixed seed and taken until the validation
    share of *rows* is reached — so a group of 400 frames does not silently
    become 40% of validation. The exact fraction is never forced: a split
    that hits 20.0% by cutting a group in half is worse than one that lands
    at 22% and keeps identities whole."""
    eligible = [r for r in rows if _get(r, "status") == Status.ACCEPTED]
    if not eligible:
        return {"train": [], "validation": [], "seed": seed,
                "algorithm": "group-shuffle", "groups": {"train": 0, "validation": 0}}

    by_group = defaultdict(list)
    for row in eligible:
        by_group[_get(row, "group_id")].append(row)

    groups = sorted(by_group)                    # sort first: order must not
    random.Random(seed).shuffle(groups)          # depend on filesystem order

    target = validation_fraction * len(eligible)
    validation_groups, taken = set(), 0
    for group in groups:
        if taken >= target:
            break
        validation_groups.add(group)
        taken += len(by_group[group])

    train_rows, validation_rows = [], []
    for group, group_rows in by_group.items():
        bucket = validation_rows if group in validation_groups else train_rows
        for row in group_rows:
            _set(row, "split", "validation" if group in validation_groups else "train")
            bucket.append(row)

    return {
        "train": train_rows,
        "validation": validation_rows,
        "seed": seed,
        "algorithm": "sort groups, shuffle with seed, take groups until the "
                     "validation row target is met; rows follow their group",
        "validation_fraction_requested": validation_fraction,
        "validation_fraction_achieved": round(len(validation_rows) / len(eligible), 4),
        "groups": {"train": len(by_group) - len(validation_groups),
                   "validation": len(validation_groups)},
    }


def assert_no_group_leakage(train_rows, validation_rows, sealed_rows=(),
                            duplicates=None):
    """Three assertions. Any failure stops the build.

    Intent is not a defence — this is the check that makes the split a fact
    rather than a hope."""
    train_groups = {_get(r, "group_id") for r in train_rows}
    validation_groups = {_get(r, "group_id") for r in validation_rows}
    sealed_groups = {_get(r, "group_id") for r in sealed_rows}

    shared = train_groups & validation_groups
    if shared:
        raise AssertionError(
            f"GROUP LEAKAGE: {len(shared)} group(s) in both train and "
            f"validation - {sorted(shared)[:10]}")

    shared = (train_groups | validation_groups) & sealed_groups
    if shared:
        raise AssertionError(
            f"SEALED LEAKAGE: {len(shared)} sealed group(s) reached "
            f"train/validation - {sorted(shared)[:10]}")

    if duplicates and duplicates.get("cross_group"):
        # These are rejected before the split reaches them. If any survived,
        # the same face is on both sides and no number from this data means
        # anything.
        survivors = {_get(r, "sample_id") for r in train_rows} | \
                    {_get(r, "sample_id") for r in validation_rows}
        leaked = [ids for ids in duplicates["cross_group"]
                  if any(i in survivors for i in ids)]
        if leaked:
            raise AssertionError(
                f"DUPLICATE ACROSS GROUPS reached a split: {len(leaked)} identical "
                "file(s) carry different group ids, so the same face sits on both "
                "sides:\n  " + "\n  ".join(str(g) for g in leaked[:5]))

    # Duplicates must not straddle a boundary even when they share a group,
    # because a group could in principle be split by a future change.
    if duplicates:
        side = {}
        for row in train_rows:
            side[_get(row, "sample_id")] = "train"
        for row in validation_rows:
            side[_get(row, "sample_id")] = "validation"
        for ids in duplicates["exact"] + duplicates["near"]:
            sides = {side[i] for i in ids if i in side}
            if len(sides) > 1:
                raise AssertionError(
                    f"DUPLICATE ACROSS SPLITS: {ids} appear in {sorted(sides)}")
    return True


# ------------------------------------------------------------- statistics

def composition(rows):
    """Counts, not recommendations. §11 and §12 read this."""
    accepted = [r for r in rows if _get(r, "status") == Status.ACCEPTED]

    def count(field):
        return dict(Counter(_get(r, field) or "(none)" for r in accepted))

    groups_by_dataset = defaultdict(set)
    for row in accepted:
        groups_by_dataset[_get(row, "dataset")].add(_get(row, "group_id"))

    total = len(accepted) or 1
    return {
        "accepted": len(accepted),
        "total_rows": len(rows),
        "by_label": count("normalized_label"),
        "by_dataset": count("dataset"),
        "by_family": count("manipulation_family"),
        "by_method": count("manipulation_method"),
        "by_compression": count("compression"),
        "by_status": dict(Counter(_get(r, "status") for r in rows)),
        "by_rejection": dict(Counter(_get(r, "rejection_reason") for r in rows
                                     if _get(r, "rejection_reason"))),
        "dataset_share": {k: round(v / total * 100, 1)
                          for k, v in count("dataset").items()},
        "groups_total": len({_get(r, "group_id") for r in accepted}),
        "groups_by_dataset": {k: len(v) for k, v in groups_by_dataset.items()},
        "faces": {
            "found": sum(1 for r in rows if _get(r, "face_found")),
            "no_face": sum(1 for r in rows
                           if _get(r, "rejection_reason") == Reject.NO_FACE),
            "multiple": sum(1 for r in rows if int(_get(r, "face_count") or 0) > 1),
        },
    }


# ------------------------------------------------------------------ helpers

def _get(row, field):
    return getattr(row, field) if hasattr(row, "dataset") else row.get(field, "")


def _set(row, field, value):
    if hasattr(row, "dataset"):
        setattr(row, field, value)
    else:
        row[field] = value
