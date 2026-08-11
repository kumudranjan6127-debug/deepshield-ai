"""The dataset pipeline, and proof that its guards actually fire.

A leakage guard that has never been seen to fail is not a guard. Every
assertion in `scripts/prepare/` is tested twice here: once that clean data
passes, and once that dirty data is **refused**. The second half is the half
that matters — a build which silently accepts Celeb-DF into training produces
numbers that look fine and mean nothing.

These tests construct their own rows. They need no datasets, no network and
no GPU.
"""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts", "prepare"))
sys.path.insert(0, os.path.join(ROOT, "training"))

pytestmark = pytest.mark.metrics   # pure-logic suite; no server, no model

from schema import (FIELDS, Family, Label, ManifestRow, Reject,  # noqa: E402
                    Status, normalise_label, read_manifest, write_manifest)
from sealed import SealError, SealedRegistry                     # noqa: E402
from split import (assert_no_group_leakage, assign_duplicate_status,  # noqa: E402
                   composition, find_duplicates, group_split)
from inventory import sample_frame_indices, stable_id            # noqa: E402


def row(sample_id, group, dataset="ffpp", label=Label.FAKE, method="deepfakes",
        status=Status.ACCEPTED, sha="", phash="", split=""):
    return ManifestRow(
        sample_id=sample_id, dataset=dataset, source_path=f"{dataset}/{sample_id}.mp4",
        group_id=group, normalized_label=label, original_label=label.lower(),
        manipulation_method=method, status=status, sha256=sha, phash=phash,
        split=split)


# ------------------------------------------------------------ manifest schema

def test_the_schema_has_every_required_field():
    required = {
        "sample_id", "dataset", "source_path", "source_id", "video_id",
        "frame_index", "timestamp", "subject_id", "group_id", "original_label",
        "normalized_label", "manipulation_method", "manipulation_family",
        "width", "height", "duration", "sha256", "phash", "face_found",
        "face_score", "face_count", "crop_box", "status", "rejection_reason",
    }
    missing = required - set(FIELDS)
    assert not missing, f"manifest schema is missing {sorted(missing)}"


def test_a_manifest_round_trips(tmp_path):
    rows = [row("a", "g1"), row("b", "g2", label=Label.REAL, method="")]
    path = tmp_path / "m.csv"
    write_manifest(path, rows)
    back = read_manifest(path)

    assert len(back) == 2
    assert back[0]["sample_id"] == "a"
    assert back[0]["group_id"] == "g1"
    assert isinstance(back[0]["face_found"], bool)
    assert isinstance(back[0]["frame_index"], int)


def test_rejected_rows_are_kept_not_deleted(tmp_path):
    """The rule the whole schema is built around."""
    rows = [row("keep", "g1"),
            row("drop", "g2", status=Status.REJECTED)]
    rows[1].rejection_reason = Reject.NO_FACE

    path = tmp_path / "m.csv"
    write_manifest(path, rows)
    back = read_manifest(path)

    assert len(back) == 2, "a rejected row disappeared from the manifest"
    dropped = [r for r in back if r["sample_id"] == "drop"][0]
    assert dropped["status"] == Status.REJECTED
    assert dropped["rejection_reason"] == Reject.NO_FACE


# --------------------------------------------------------- label normalisation

@pytest.mark.parametrize("original,method,label,family", [
    ("real", "", Label.REAL, Family.REAL),
    ("original", "", Label.REAL, Family.REAL),
    ("Celeb-real", "", Label.REAL, Family.REAL),
    ("fake", "deepfakes", Label.FAKE, Family.FACE_SWAP),
    ("Deepfakes", "deepfakes", Label.FAKE, Family.FACE_SWAP),
    ("fake", "face2face", Label.FAKE, Family.FACE_REENACTMENT),
    ("fake", "neuraltextures", Label.FAKE, Family.FACE_REENACTMENT),
    ("fake", "stylegan2", Label.FAKE, Family.GAN),
    ("fake", "diffusion", Label.FAKE, Family.DIFFUSION),
    ("celeb-synthesis", "celeb-synthesis", Label.FAKE, Family.FACE_SWAP),
])
def test_labels_normalise_without_losing_the_original(original, method, label, family):
    normalized, resolved = normalise_label(original, method)
    assert normalized == label
    assert resolved == family


def test_an_unknown_label_is_unknown_not_guessed():
    """A gap in the mapping table must look like a gap."""
    label, family = normalise_label("mystery_manipulation", "")
    assert label == Label.UNKNOWN


def test_an_unknown_method_gives_unknown_fake():
    label, family = normalise_label("fake", "some_new_generator_2027")
    assert label == Label.FAKE
    assert family == Family.UNKNOWN_FAKE, "a manipulation family was invented"


# ---------------------------------------------------------------- determinism

def test_sample_ids_are_stable():
    assert stable_id("ffpp", "a/b.mp4", 5) == stable_id("ffpp", "a/b.mp4", 5)
    assert stable_id("ffpp", "a/b.mp4", 5) != stable_id("ffpp", "a/b.mp4", 6)
    assert stable_id("ffpp", "a/b.mp4") != stable_id("dfdc", "a/b.mp4")


def test_frame_sampling_is_deterministic_and_uniform():
    a = sample_frame_indices(300, 30.0, 8)
    b = sample_frame_indices(300, 30.0, 8)
    assert a == b, "frame selection is not reproducible"
    assert len(a) == 8
    assert a == sorted(a)
    assert a[0] > 0, "sampling should not start at frame 0"
    assert a[-1] < 300

    gaps = [j - i for i, j in zip(a, a[1:])]
    assert max(gaps) - min(gaps) <= 1, f"sampling is not uniform: {gaps}"


def test_frame_sampling_handles_short_and_empty_clips():
    assert sample_frame_indices(5, 25.0, 8) == list(range(5))
    assert sample_frame_indices(0, 25.0, 8) == []
    assert sample_frame_indices(300, 30.0, 0) == []


def test_the_split_is_deterministic():
    rows = [row(f"s{i}", f"g{i // 3}") for i in range(60)]
    a = group_split(list(rows), 0.2, seed=42)
    b = group_split(list(rows), 0.2, seed=42)
    c = group_split(list(rows), 0.2, seed=7)

    assert {r.sample_id for r in a["train"]} == {r.sample_id for r in b["train"]}
    assert {r.sample_id for r in a["train"]} != {r.sample_id for r in c["train"]}
    assert a["seed"] == 42


def test_the_split_records_how_it_split():
    info = group_split([row(f"s{i}", f"g{i}") for i in range(50)], 0.2, seed=42)
    assert info["seed"] == 42
    assert "shuffle" in info["algorithm"]
    assert 0.0 < info["validation_fraction_achieved"] < 0.5


# ------------------------------------------------------------- group leakage

def test_a_group_never_spans_train_and_validation():
    rows = [row(f"s{i}", f"g{i // 5}") for i in range(100)]
    info = group_split(rows, 0.2, seed=42)

    train_groups = {r.group_id for r in info["train"]}
    validation_groups = {r.group_id for r in info["validation"]}
    assert not (train_groups & validation_groups)
    assert_no_group_leakage(info["train"], info["validation"])


def test_group_leakage_is_refused():
    """Hand-built leakage: the guard must catch it."""
    train = [row("a", "shared_identity")]
    validation = [row("b", "shared_identity")]
    with pytest.raises(AssertionError, match="GROUP LEAKAGE"):
        assert_no_group_leakage(train, validation)


def test_sealed_groups_reaching_a_split_are_refused():
    train = [row("a", "g1")]
    validation = [row("b", "g2")]
    sealed = [row("c", "g1", dataset="celebdf")]
    with pytest.raises(AssertionError, match="SEALED LEAKAGE"):
        assert_no_group_leakage(train, validation, sealed)


def test_a_duplicate_across_splits_is_refused():
    train = [row("a", "g1", sha="deadbeef")]
    validation = [row("b", "g2", sha="deadbeef")]
    duplicates = {"exact": [["a", "b"]], "near": [], "cross_group": []}
    with pytest.raises(AssertionError, match="DUPLICATE ACROSS SPLITS"):
        assert_no_group_leakage(train, validation, [], duplicates)


# ------------------------------------------------------------ sealed protection

@pytest.fixture
def registry(tmp_path):
    return SealedRegistry(groups_file=tmp_path / "sealed.txt")


@pytest.mark.parametrize("dirty,why", [
    (lambda: row("x", "g", dataset="celebdf"), "Celeb-DF"),
    (lambda: row("x", "g", dataset="celebdf_v2"), "a renamed Celeb-DF folder"),
    (lambda: row("x", "g", dataset="deeperforensics"), "DeeperForensics"),
    (lambda: row("x", "g", method="neuraltextures"), "the withheld FF++ method"),
])
def test_sealed_data_is_refused_in_training(registry, dirty, why):
    with pytest.raises(SealError, match="SEALED DATA"):
        registry.assert_clean([dirty()], "train manifest")


def test_a_sealed_group_is_refused_even_under_a_different_dataset_name(registry):
    """The layer that survives a rename: the committed group list."""
    registry.record_groups(["celebdf:id42"])
    disguised = row("x", "celebdf:id42", dataset="totally_fine_dataset",
                    method="deepfakes")
    with pytest.raises(SealError):
        registry.assert_clean([disguised], "train manifest")


def test_clean_training_data_passes(registry):
    assert registry.assert_clean([row("a", "ffpp:1"), row("b", "dfdc:2",
                                                          dataset="dfdc")])


def test_partition_seals_rather_than_deletes(registry):
    rows = [row("keep", "g1"), row("seal", "g2", dataset="celebdf")]
    trainable, sealed = registry.partition(rows)

    assert len(trainable) == 1 and len(sealed) == 1
    assert sealed[0].status == Status.SEALED
    assert "celebdf" in sealed[0].notes
    assert len(trainable) + len(sealed) == len(rows), "a row was dropped"


def test_the_seal_is_additive(tmp_path):
    """Un-sealing must be a deliberate edit, not a re-run."""
    path = tmp_path / "sealed.txt"
    first = SealedRegistry(groups_file=path)
    first.record_groups(["a", "b"])

    second = SealedRegistry(groups_file=path)
    second.record_groups(["c"])

    third = SealedRegistry(groups_file=path)
    assert third.sealed_groups == {"a", "b", "c"}, "a re-run un-sealed data"


def test_the_seal_file_survives_outside_the_gitignored_tree():
    from sealed import SEALED_GROUPS_FILE
    relative = os.path.relpath(SEALED_GROUPS_FILE, ROOT).replace(os.sep, "/")
    assert relative.startswith("docs/"), \
        "the seal lives inside datasets/, which is gitignored and cleanable"


def test_the_training_manifest_contains_only_allowed_datasets(registry):
    allowed = {"ffpp", "dfdc", "generated", "lfw", "phone", "local_clips"}
    rows = [row("a", "g1", dataset="ffpp"), row("b", "g2", dataset="dfdc")]
    registry.assert_clean(rows)
    assert all(r.dataset in allowed for r in rows)


# ------------------------------------------------------------------ duplicates

def test_exact_duplicates_are_found_and_marked():
    rows = [row("a", "g1", sha="same"), row("b", "g1", sha="same"),
            row("c", "g2", sha="other")]
    duplicates = find_duplicates(rows)

    assert len(duplicates["exact"]) == 1
    assert set(duplicates["exact"][0]) == {"a", "b"}

    marked = assign_duplicate_status(rows, duplicates)
    assert marked == 1, "one copy should survive, one should be marked"
    assert sum(1 for r in rows if r.status == Status.ACCEPTED) == 2


def test_a_cross_group_duplicate_rejects_every_copy():
    """The same file under two identities: neither assignment is trustworthy,
    so no copy may be used. Found in real data on the first run — one LFW
    photograph appears under two different people's names."""
    rows = [row("a", "person_one", sha="same"),
            row("b", "person_two", sha="same"),
            row("c", "person_three", sha="different")]
    duplicates = find_duplicates(rows)
    assert len(duplicates["cross_group"]) == 1

    assign_duplicate_status(rows, duplicates)
    statuses = {r.sample_id: (r.status, r.rejection_reason) for r in rows}
    assert statuses["a"] == (Status.REJECTED, Reject.UNSAFE_GROUP)
    assert statuses["b"] == (Status.REJECTED, Reject.UNSAFE_GROUP)
    assert statuses["c"][0] == Status.ACCEPTED

    # ...and having been rejected, they cannot reach a split
    info = group_split(rows, 0.2, seed=42)
    assert_no_group_leakage(info["train"], info["validation"], [], duplicates)


def test_near_duplicates_are_only_sought_within_a_group():
    """Two different people may hash close together; dropping those would
    lose real diversity."""
    rows = [row("a", "g1", phash="ffffffffffffffff"),
            row("b", "g2", phash="ffffffffffffffff")]   # same hash, other person
    duplicates = find_duplicates(rows)
    assert duplicates["near"] == [], "a near-duplicate was found across groups"


def test_near_duplicates_inside_a_group_are_marked():
    rows = [row("a", "g1", phash="ffffffffffffffff"),
            row("b", "g1", phash="ffffffffffffffff")]
    duplicates = find_duplicates(rows)
    assert len(duplicates["near"]) == 1
    assign_duplicate_status(rows, duplicates)
    assert sum(1 for r in rows if r.status == Status.ACCEPTED) == 1


# ------------------------------------------------------- missing metadata

def test_a_sample_with_no_group_is_unsafe_not_guessed():
    """When provenance cannot establish an identity, the sample is refused.
    Guessing is how a face ends up on both sides."""
    from inventory import inventory_directory
    import tempfile
    from PIL import Image
    import numpy as np

    with tempfile.TemporaryDirectory() as tmp:
        Image.fromarray(np.zeros((64, 64, 3), "uint8")).save(
            os.path.join(tmp, "mystery.png"))
        rows = inventory_directory(tmp, "test",
                                   label_of=lambda p: "real",
                                   group_of=lambda p: None)      # no group
    assert len(rows) == 1
    assert rows[0].status == Status.REJECTED
    assert rows[0].rejection_reason == Reject.UNSAFE_GROUP


def test_an_unmapped_label_is_rejected_not_defaulted():
    from inventory import inventory_directory
    import tempfile
    from PIL import Image
    import numpy as np

    with tempfile.TemporaryDirectory() as tmp:
        Image.fromarray(np.zeros((64, 64, 3), "uint8")).save(
            os.path.join(tmp, "x.png"))
        rows = inventory_directory(tmp, "test",
                                   label_of=lambda p: "who_knows",
                                   group_of=lambda p: "g1")
    assert rows[0].status == Status.REJECTED
    assert rows[0].rejection_reason == Reject.INVALID_LABEL


# --------------------------------------------------------------- no-face

def test_a_faceless_image_is_recorded_not_silently_dropped(no_face_image, tmp_path):
    from inventory import FaceExtractor
    yunet = os.path.join(ROOT, "models", "face_detection_yunet.onnx")
    if not os.path.exists(yunet):
        pytest.skip("YuNet not present")

    extractor = FaceExtractor(yunet, str(tmp_path))
    sample = row("x", "g1", dataset="test")
    sample.source_path = os.path.relpath(no_face_image, ROOT).replace(os.sep, "/")
    sample.status = Status.PENDING

    extractor.extract(sample)
    assert sample.status == Status.REJECTED
    assert sample.rejection_reason == Reject.NO_FACE
    assert sample.face_found is False
    assert sample.notes, "no-face samples should say why they are kept"


def test_a_face_is_cropped_through_the_production_path(fake_face, tmp_path):
    from inventory import FaceExtractor
    yunet = os.path.join(ROOT, "models", "face_detection_yunet.onnx")
    if not os.path.exists(yunet):
        pytest.skip("YuNet not present")

    extractor = FaceExtractor(yunet, str(tmp_path))
    sample = row("x", "g1", dataset="test")
    sample.source_path = os.path.relpath(fake_face, ROOT).replace(os.sep, "/")
    sample.status = Status.PENDING

    extractor.extract(sample)
    assert sample.status == Status.ACCEPTED
    assert sample.face_found is True
    assert sample.crop_path and os.path.exists(os.path.join(ROOT, sample.crop_path))
    assert sample.crop_path.endswith(".png"), "crops must be lossless"
    assert sample.phash and sample.crop_box


def test_the_extractor_does_not_reimplement_face_detection():
    """Requirement: call the pinned module, do not copy it a third time."""
    import re
    source = open(os.path.join(ROOT, "scripts", "prepare", "inventory.py"),
                  encoding="utf-8").read()
    assert "from deepshield_preprocess import Preprocessor" in source
    code = re.sub(r'"""[\s\S]*?"""', "", source)
    assert "FaceDetectorYN_create" not in code, \
        "face detection was reimplemented instead of imported"


# ------------------------------------------------------------ raw data hygiene

def test_raw_datasets_are_never_committed():
    import subprocess
    tracked = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True,
                             text=True).stdout.splitlines()
    offenders = [p for p in tracked if p.startswith("datasets/")]
    assert not offenders, f"dataset files are tracked in git: {offenders[:10]}"


def test_the_datasets_tree_is_gitignored():
    ignored = open(os.path.join(ROOT, ".gitignore"), encoding="utf-8").read()
    assert "datasets/" in ignored


# ------------------------------------------------------------------ statistics

def test_composition_counts_groups_not_just_rows():
    rows = [row(f"s{i}", "one_group") for i in range(50)]
    stats = composition(rows)
    assert stats["accepted"] == 50
    assert stats["groups_total"] == 1, \
        "50 frames of one identity is one independent sample, not 50"


def test_composition_reports_dataset_share():
    rows = ([row(f"a{i}", f"g{i}", dataset="dfdc") for i in range(90)]
            + [row(f"b{i}", f"h{i}", dataset="ffpp") for i in range(10)])
    stats = composition(rows)
    assert stats["dataset_share"]["dfdc"] == 90.0
    assert stats["dataset_share"]["ffpp"] == 10.0
