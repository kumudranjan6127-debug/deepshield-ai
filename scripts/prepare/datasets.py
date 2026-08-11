"""Per-dataset adapters: label, group, method, compression.

Each dataset knows its own layout and nothing about the others. An adapter
returns nothing when its data is absent, so the pipeline runs end to end on
whatever is actually present rather than failing on what is not.

------------------------------------------------------------ the group rule

Grouping is where this gets subtle, and getting it wrong is invisible.

FaceForensics++ names a manipulation `033_097.mp4` — the face from 097 on the
target 033. Both originals must sit on the same side of the split as that
manipulation, **and a video appears in several pairs**: 033 might also be
paired with 112, which is paired with 220. Splitting on either id alone puts
the same face on both sides.

The correct unit is the **connected component** of the pairing graph. Union
every id that shares a manipulation, and the component is the group. Celeb-DF
has the same structure (`id0_id1_0000.mp4`) and gets the same treatment.

Where the metadata cannot establish a group, the adapter returns `None` and
the sample is rejected as `UNSAFE_GROUP`. Guessing is how an identity ends up
on both sides.
"""
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW = os.path.join(ROOT, "datasets", "raw")

__all__ = ["UnionFind", "ADAPTERS", "adapter_for", "available"]


class UnionFind:
    """Connected components over video ids linked by manipulations."""

    def __init__(self):
        self.parent = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)   # deterministic root

    def component(self, x):
        return self.find(x)


# ---------------------------------------------------------------- adapters

class Adapter:
    """Base: a dataset that is not present contributes nothing, quietly."""
    name = "unknown"
    root = ""
    requires_approval = True

    def __init__(self, root=None):
        self.root = root or os.path.join(RAW, self.name)

    def present(self):
        return os.path.isdir(self.root) and any(os.scandir(self.root))

    def prepare(self):
        """Read whatever index the dataset ships before walking files."""

    def label_of(self, path):
        raise NotImplementedError

    def group_of(self, path):
        raise NotImplementedError

    def method_of(self, path):
        return ""

    def compression_of(self, path):
        return "unknown"

    def subject_of(self, path):
        return ""


class FaceForensics(Adapter):
    """original_sequences/... and manipulated_sequences/<Method>/<c>/videos/*.mp4"""
    name = "ffpp"

    METHODS = ("Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures", "FaceShifter")
    PAIR = re.compile(r"^(\d+)_(\d+)$")

    def prepare(self):
        """Union every id that shares a manipulation, once, up front."""
        self.uf = UnionFind()
        manipulated = os.path.join(self.root, "manipulated_sequences")
        if not os.path.isdir(manipulated):
            return
        for dirpath, _, filenames in os.walk(manipulated):
            for name in filenames:
                match = self.PAIR.match(os.path.splitext(name)[0])
                if match:
                    self.uf.union(match.group(1), match.group(2))

    def _ids(self, path):
        stem = os.path.splitext(os.path.basename(path))[0]
        match = self.PAIR.match(stem)
        return (match.group(1), match.group(2)) if match else (stem, None)

    def label_of(self, path):
        return "fake" if "manipulated_sequences" in path.replace("\\", "/") else "real"

    def method_of(self, path):
        parts = path.replace("\\", "/").split("/")
        for method in self.METHODS:
            if method in parts:
                return method.lower()
        return ""

    def compression_of(self, path):
        for level in ("raw", "c23", "c40"):
            if f"/{level}/" in path.replace("\\", "/"):
                return level
        return "unknown"

    def group_of(self, path):
        first, second = self._ids(path)
        if not first.isdigit():
            return None                      # unexpected name — do not guess
        component = self.uf.component(first) if hasattr(self, "uf") else first
        if second:
            component = min(component, self.uf.component(second))
        return f"ffpp:{component}"


class DFDC(Adapter):
    """Folders of .mp4 with a metadata.json naming each clip's original."""
    name = "dfdc"

    def prepare(self):
        self.meta = {}
        for dirpath, _, filenames in os.walk(self.root):
            if "metadata.json" not in filenames:
                continue
            try:
                with open(os.path.join(dirpath, "metadata.json"), encoding="utf-8") as f:
                    entries = json.load(f)
            except Exception:
                continue
            for clip, info in entries.items():
                self.meta[clip] = {
                    "label": str(info.get("label", "")).lower(),
                    "original": info.get("original") or clip,
                }

    def _entry(self, path):
        return getattr(self, "meta", {}).get(os.path.basename(path))

    def label_of(self, path):
        entry = self._entry(path)
        return entry["label"] if entry else ""

    def method_of(self, path):
        return "dfdc" if (self._entry(path) or {}).get("label") == "fake" else ""

    def group_of(self, path):
        entry = self._entry(path)
        if not entry:
            return None                      # not in metadata — unsafe to split
        return "dfdc:" + os.path.splitext(entry["original"])[0]


class CelebDF(Adapter):
    """SEALED. Celeb-real/id0_0000.mp4, Celeb-synthesis/id0_id1_0000.mp4"""
    name = "celebdf"

    PAIR = re.compile(r"^id(\d+)_id(\d+)_\d+$")
    SINGLE = re.compile(r"^id(\d+)_\d+$")

    def prepare(self):
        self.uf = UnionFind()
        for dirpath, _, filenames in os.walk(self.root):
            for name in filenames:
                match = self.PAIR.match(os.path.splitext(name)[0])
                if match:
                    self.uf.union(match.group(1), match.group(2))

    def label_of(self, path):
        folder = os.path.basename(os.path.dirname(path)).lower()
        if "synthesis" in folder:
            return "celeb-synthesis"
        if "real" in folder:
            return "real"
        return ""

    def method_of(self, path):
        return "celeb-synthesis" if "synthesis" in path.lower() else ""

    def group_of(self, path):
        stem = os.path.splitext(os.path.basename(path))[0]
        pair = self.PAIR.match(stem)
        if pair:
            return "celebdf:" + min(self.uf.component(pair.group(1)),
                                    self.uf.component(pair.group(2)))
        single = self.SINGLE.match(stem)
        if single:
            return "celebdf:" + self.uf.component(single.group(1))
        if "youtube" in path.lower():
            return "celebdf:youtube:" + stem     # distinct source, its own group
        return None


class DeeperForensics(Adapter):
    """SEALED. Source videos organised by actor."""
    name = "deeperforensics"

    ACTOR = re.compile(r"(?:^|[_/])(M\d{3}|W\d{3})", re.I)

    def label_of(self, path):
        lowered = path.lower().replace("\\", "/")
        if "manipulated" in lowered or "/end_to_end" in lowered:
            return "fake"
        if "source" in lowered or "/original" in lowered:
            return "real"
        return ""

    def method_of(self, path):
        return "deeperforensics" if self.label_of(path) == "fake" else ""

    def group_of(self, path):
        match = self.ACTOR.search(path.replace("\\", "/"))
        return f"deeperforensics:{match.group(1).upper()}" if match else None

    def subject_of(self, path):
        match = self.ACTOR.search(path.replace("\\", "/"))
        return match.group(1).upper() if match else ""


class PhonePhotos(Adapter):
    """Consented phone photographs. `groups.csv` (path,group) is required.

    Without it there is no way to know two photographs are of the same person
    or from the same device, so every sample is rejected as UNSAFE_GROUP
    rather than treated as independent."""
    name = "phone"

    def prepare(self):
        self.groups = {}
        index = os.path.join(self.root, "groups.csv")
        if not os.path.exists(index):
            return
        import csv
        with open(index, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("path"):
                    key = os.path.normcase(os.path.basename(row["path"]))
                    self.groups[key] = row.get("group") or ""

    def label_of(self, path):
        return "real"

    def group_of(self, path):
        group = getattr(self, "groups", {}).get(os.path.normcase(os.path.basename(path)))
        return f"phone:{group}" if group else None

    def compression_of(self, path):
        return "jpeg"


class GeneratedFaces(Adapter):
    """Fully-synthetic stills. One image is one independent draw, so one
    image is one group — no identity is shared between StyleGAN samples."""
    name = "generated"

    def label_of(self, path):
        return "fake"

    def method_of(self, path):
        lowered = path.lower().replace("\\", "/")
        for key in ("stylegan2", "stylegan", "tpdn", "diffusion", "stable-diffusion"):
            if key in lowered:
                return key
        return ""

    def group_of(self, path):
        return "generated:" + os.path.splitext(os.path.basename(path))[0]

    def compression_of(self, path):
        return "jpeg"


class LFW(Adapter):
    """Real press photographs. The person's name is in the filename, so the
    group is free and correct."""
    name = "lfw"
    requires_approval = False

    def label_of(self, path):
        return "real"

    def group_of(self, path):
        stem = os.path.splitext(os.path.basename(path))[0]
        stem = stem.split("__")[0]                     # our own suffix
        parts = stem.rsplit("_", 1)
        person = parts[0] if len(parts) == 2 and parts[1].isdigit() else stem
        return f"lfw:{person}"

    def subject_of(self, path):
        return self.group_of(path).split(":", 1)[1]

    def compression_of(self, path):
        return "jpeg"


class LocalClips(Adapter):
    """The two committed test clips. One clip is one group."""
    name = "local_clips"
    requires_approval = False

    def label_of(self, path):
        stem = os.path.basename(path).lower()
        if "authentic" in stem or "real" in stem:
            return "real"
        if "synthetic" in stem or "fake" in stem:
            return "fake"
        return ""

    def method_of(self, path):
        return "stylegan2" if self.label_of(path) == "fake" else ""

    def group_of(self, path):
        return "local_clips:" + os.path.splitext(os.path.basename(path))[0]


ADAPTERS = {
    a.name: a for a in (FaceForensics, DFDC, CelebDF, DeeperForensics,
                        PhonePhotos, GeneratedFaces, LFW, LocalClips)
}


def adapter_for(name, root=None):
    if name not in ADAPTERS:
        raise KeyError(f"unknown dataset {name!r}; known: {sorted(ADAPTERS)}")
    adapter = ADAPTERS[name](root)
    adapter.prepare()
    return adapter


def available(roots=None):
    """Which adapters have data on this machine right now."""
    found = {}
    for name in ADAPTERS:
        root = (roots or {}).get(name)
        adapter = ADAPTERS[name](root)
        found[name] = adapter.present()
    return found
