"""Build the V4 dataset: inventory → extract → dedupe → seal → split → report.

    python scripts/prepare/build.py --list
    python scripts/prepare/build.py --dry-run
    python scripts/prepare/build.py --datasets lfw generated local_clips
    python scripts/prepare/build.py --all --frames 8

Nothing is downloaded and nothing is trained. The builder reads what is
already under `datasets/raw/`, and datasets that are absent contribute
nothing rather than causing a failure.

Order matters and is not negotiable:

    1. inventory   every source file becomes a row
    2. extract     frames sampled, faces cropped through the production path
    3. dedupe      exact and near, reported, never deleted
    4. seal        sealed rows partitioned out and the seal recorded
    5. GUARD       assert no sealed sample survived into the trainable set
    6. split       groups shuffled with a fixed seed, rows follow their group
    7. GUARD       assert no group and no duplicate crosses a boundary
    8. report      counts, and everything that was rejected, with reasons

Step 5 runs before the split, and step 7 after it. There is no flag to skip
either one.
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "training"))

import datasets as ds                                       # noqa: E402
from inventory import FaceExtractor, expand_video_rows, inventory_directory, sha256_of  # noqa: E402
from schema import Status, read_manifest, write_manifest    # noqa: E402
from sealed import SealedRegistry                           # noqa: E402
from split import (assert_no_group_leakage, assign_duplicate_status,  # noqa: E402
                   composition, find_duplicates, group_split)

DATA = os.path.join(ROOT, "datasets")
MANIFESTS = os.path.join(DATA, "manifests")
CROPS = os.path.join(DATA, "crops")
REPORTS = os.path.join(DATA, "reports")
YUNET = os.path.join(ROOT, "models", "face_detection_yunet.onnx")

# Where each adapter looks when its data is not under datasets/raw/<name>.
# These point at material already committed to the repository, so the
# pipeline can be exercised end to end without downloading anything.
LOCAL_ROOTS = {
    "generated": os.path.join(ROOT, "training", "tpdn_test"),
    "lfw": os.path.join(ROOT, "eval_data", "real", "lfw"),
    "local_clips": os.path.join(ROOT, "training", "video_test"),
}


def log(message):
    print(message, flush=True)


# ------------------------------------------------------------------- stages

def stage_inventory(names, frames_per_video, limit=None):
    rows = []
    for name in names:
        root = LOCAL_ROOTS.get(name) or os.path.join(DATA, "raw", name)
        adapter = ds.adapter_for(name, root)
        if not adapter.present():
            log(f"  {name:18s} absent - contributes nothing")
            continue

        found = inventory_directory(
            root, name,
            label_of=adapter.label_of, group_of=adapter.group_of,
            method_of=adapter.method_of, compression_of=adapter.compression_of,
            subject_of=adapter.subject_of, limit=limit)

        expanded = []
        for row in found:
            if row.media_type == "video" and row.status != Status.REJECTED:
                children = expand_video_rows(row, frames_per_video)
                expanded.append(row)          # the parent record is kept
                row.status = Status.REJECTED if not children else row.status
                if children:
                    row.notes = f"expanded into {len(children)} sampled frames"
                    row.rejection_reason = ""
                    row.status = Status.PENDING
                    row.split = "parent"
                expanded.extend(children)
            else:
                expanded.append(row)

        videos = sum(1 for r in found if r.media_type == "video")
        log(f"  {name:18s} {len(found):6d} source files "
            f"({videos} video) -> {len(expanded)} rows")
        rows.extend(expanded)
    return rows


def stage_extract(rows, limit=None):
    """Crop through the production path. Parent video rows are skipped."""
    extractor = FaceExtractor(YUNET, CROPS)
    targets = [r for r in rows if r.status == Status.PENDING and r.split != "parent"]
    if limit:
        targets = targets[:limit]

    for i, row in enumerate(targets, 1):
        source = os.path.join(ROOT, row.source_path)
        if row.frame_index < 0 and os.path.exists(source):
            row.sha256 = sha256_of(source)
        extractor.extract(row)
        if i % 50 == 0 or i == len(targets):
            log(f"\r  extracted {i}/{len(targets)}")
    return rows


def stage_report(rows, split_info, duplicates, registry, stats, path):
    """docs/V4_DATASET_REPORT.md — generated, never hand-typed."""
    accepted = stats["accepted"]

    def table(title, mapping, total=None):
        if not mapping:
            return f"**{title}:** none\n"
        lines = [f"| {title} | Count | Share |", "|---|---|---|"]
        denominator = total or sum(mapping.values()) or 1
        for key, value in sorted(mapping.items(), key=lambda kv: -kv[1]):
            lines.append(f"| `{key}` | {value:,} | {value / denominator * 100:.1f}% |")
        return "\n".join(lines) + "\n"

    seal = registry.describe()
    lines = [
        "# V4 Dataset Report",
        "",
        "> Generated by `python scripts/prepare/build.py`. Every number below "
        "is a count of rows in `datasets/manifests/all.csv`. **No model was "
        "trained and no dataset was downloaded to produce it.**",
        "",
        "## 1. Totals",
        "",
        f"| | |",
        f"|---|---|",
        f"| Manifest rows (nothing is ever deleted) | **{stats['total_rows']:,}** |",
        f"| Accepted - eligible for train/validation | **{accepted:,}** |",
        f"| Sealed - evaluation only | **{stats['by_status'].get(Status.SEALED, 0):,}** |",
        f"| Rejected - with a reason | **{stats['by_status'].get(Status.REJECTED, 0):,}** |",
        f"| Independent groups (the honest sample size) | **{stats['groups_total']:,}** |",
        "",
        "## 2. Labels",
        "",
        table("Label", stats["by_label"]),
        "## 3. Datasets",
        "",
        table("Dataset", stats["by_dataset"]),
        "**Dataset share of the accepted pool** — no single source may quietly "
        "dominate:",
        "",
        table("Share", stats["dataset_share"], total=100),
        "## 4. Manipulation",
        "",
        table("Family", stats["by_family"]),
        table("Method", stats["by_method"]),
        "## 5. Compression",
        "",
        table("Compression", stats["by_compression"]),
        "## 6. Face detection",
        "",
        f"| | |",
        f"|---|---|",
        f"| Faces found | {stats['faces']['found']:,} |",
        f"| No face - kept for inspection, never trained on | {stats['faces']['no_face']:,} |",
        f"| More than one face in frame | {stats['faces']['multiple']:,} |",
        "",
        "## 7. Rejections",
        "",
        "Every rejected sample keeps its manifest row and its reason, so the "
        "corpus stays reconstructible.",
        "",
        table("Reason", stats["by_rejection"]),
        "## 8. Duplicates",
        "",
        f"| | |",
        f"|---|---|",
        f"| Exact duplicate sets (SHA-256) | {len(duplicates['exact']):,} |",
        f"| Near-duplicate pairs (dHash <= 4, within group) | {len(duplicates['near']):,} |",
        f"| **Duplicates spanning two groups** | **{len(duplicates['cross_group']):,}** |",
        "",
        "A duplicate spanning two groups means the group key is wrong and no "
        "split can be trusted; the build stops if any is found.",
        "",
        "## 9. Split",
        "",
        f"| | |",
        f"|---|---|",
        f"| Algorithm | {split_info['algorithm']} |",
        f"| Seed | `{split_info['seed']}` |",
        f"| Train rows | {len(split_info['train']):,} |",
        f"| Validation rows | {len(split_info['validation']):,} |",
        f"| Train groups | {split_info['groups']['train']:,} |",
        f"| Validation groups | {split_info['groups']['validation']:,} |",
        f"| Validation share requested | {split_info.get('validation_fraction_requested', 0):.0%} |",
        f"| Validation share achieved | {split_info.get('validation_fraction_achieved', 0):.1%} |",
        "",
        "The exact percentage is not forced. A split that hits 20.0% by "
        "cutting a group in half is worse than one that lands at 22% and "
        "keeps identities whole.",
        "",
        "## 10. Seal",
        "",
        f"| | |",
        f"|---|---|",
        f"| Sealed datasets | {', '.join(seal['sealed_datasets']) or 'none present'} |",
        f"| Withheld FF++ method | `{seal['withheld_ffpp_method']}` |",
        f"| Sealed groups recorded | {seal['sealed_groups']:,} |",
        f"| Seal file | `{seal['groups_file']}` |",
        "",
        "## 11. Reproducibility",
        "",
        "```bash",
        "python scripts/prepare/build.py --all --frames 8",
        "```",
        "",
        "Deterministic by construction: frame indices are uniform rather than "
        "random, `sample_id` is a hash of dataset + path + frame index, and "
        f"the split shuffles a *sorted* group list with seed `{split_info['seed']}` "
        "so filesystem order cannot influence it.",
        "",
        "## 12. What is still unknown",
        "",
        "- Any dataset marked `absent` above contributed nothing to these "
        "counts. See `V4_DATASET_PROVENANCE.md` for access status.",
        "- Samples rejected as `UNSAFE_GROUP` had no group key in their "
        "metadata. They are recorded, not guessed at.",
        "- `manipulation_family` is `UNKNOWN_FAKE` wherever the dataset does "
        "not state the method.",
        "",
    ]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines))
    return path


# --------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--datasets", nargs="*", help="which adapters to run")
    ap.add_argument("--all", action="store_true", help="every adapter present")
    ap.add_argument("--frames", type=int, default=8, help="frames per video")
    ap.add_argument("--validation", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, help="cap files per dataset (smoke test)")
    ap.add_argument("--list", action="store_true", help="show what is present")
    ap.add_argument("--dry-run", action="store_true", help="inventory only")
    args = ap.parse_args()

    present = ds.available(LOCAL_ROOTS)
    if args.list:
        log("Dataset adapters:")
        for name, found in sorted(present.items()):
            adapter = ds.ADAPTERS[name]
            log(f"  {name:18s} {'PRESENT' if found else 'absent ':8s} "
                f"{'(approval required)' if adapter.requires_approval else ''}")
        return 0

    names = args.datasets or [n for n, p in present.items() if p]
    if not names:
        log("no datasets present - nothing to build")
        return 0

    registry = SealedRegistry()
    log(f"DeepShield V4 dataset build - seed {args.seed}")
    log(f"  seal: datasets {registry.sealed_datasets}, "
        f"withheld FF++ method {registry.withheld!r}")

    log("\n[1/8] inventory")
    rows = stage_inventory(names, args.frames, args.limit)
    if not rows:
        log("nothing inventoried")
        return 0

    if args.dry_run:
        write_manifest(os.path.join(MANIFESTS, "inventory.csv"), rows)
        log(f"\ndry run: {len(rows)} rows -> datasets/manifests/inventory.csv")
        return 0

    log("\n[2/8] extract faces (production path)")
    rows = stage_extract(rows)

    log("\n[3/8] duplicates")
    duplicates = find_duplicates(rows)
    marked = assign_duplicate_status(rows, duplicates)
    log(f"  {len(duplicates['exact'])} exact sets, {len(duplicates['near'])} near "
        f"pairs, {len(duplicates['cross_group'])} spanning groups -> {marked} rows marked")

    log("\n[4/8] seal")
    trainable, sealed_rows = registry.partition(rows)
    if sealed_rows:
        registry.record_groups({r.group_id for r in sealed_rows if r.group_id})
    log(f"  {len(sealed_rows)} sealed, {len(trainable)} trainable")

    log("\n[5/8] GUARD: no sealed data in the trainable pool")
    registry.assert_clean(trainable, "trainable pool")
    log("  clean")

    log("\n[6/8] split by group")
    split_info = group_split(trainable, args.validation, args.seed)
    log(f"  train {len(split_info['train'])} rows / {split_info['groups']['train']} groups")
    log(f"  validation {len(split_info['validation'])} rows / "
        f"{split_info['groups']['validation']} groups")

    log("\n[7/8] GUARD: no group or duplicate crosses a boundary")
    assert_no_group_leakage(split_info["train"], split_info["validation"],
                            sealed_rows, duplicates)
    registry.assert_clean(split_info["train"], "train manifest")
    registry.assert_clean(split_info["validation"], "validation manifest")
    log("  clean")

    log("\n[8/8] manifests and report")
    os.makedirs(MANIFESTS, exist_ok=True)
    write_manifest(os.path.join(MANIFESTS, "all.csv"), rows)
    write_manifest(os.path.join(MANIFESTS, "train.csv"), split_info["train"])
    write_manifest(os.path.join(MANIFESTS, "validation.csv"), split_info["validation"])
    write_manifest(os.path.join(MANIFESTS, "sealed.csv"), sealed_rows)
    write_manifest(os.path.join(MANIFESTS, "rejected.csv"),
                   [r for r in rows if r.status == Status.REJECTED])

    stats = composition(rows)
    report = stage_report(rows, split_info, duplicates, registry, stats,
                          os.path.join(ROOT, "docs", "V4_DATASET_REPORT.md"))

    os.makedirs(REPORTS, exist_ok=True)
    import json
    with open(os.path.join(REPORTS, "duplicate_report.json"), "w",
              encoding="utf-8", newline="\n") as f:
        json.dump({"exact": duplicates["exact"], "near": duplicates["near"],
                   "cross_group": duplicates["cross_group"],
                   "near_distance_threshold": 4}, f, indent=2)

    log(f"  manifests -> datasets/manifests/")
    log(f"  report    -> {os.path.relpath(report, ROOT)}")
    log(f"\naccepted {stats['accepted']} of {stats['total_rows']} rows, "
        f"{stats['groups_total']} groups")
    log("NO MODEL WAS TRAINED. NO DATASET WAS DOWNLOADED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
