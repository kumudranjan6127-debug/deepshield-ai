"""
============================================================
DeepShield AI — real inference engine (Phase 4)

Loads models/deepshield_mobilenetv3.pth (produced by
training/DeepShield_Training_Colab.ipynb) and scores images
and videos on CPU.

Design:
- Zero hard dependency at import time — torch/cv2 are imported
  lazily, so app.py works (echo mode) even before Phase 4 deps
  are installed or the checkpoint exists.
- engine_available() is the single switch app.py checks.
============================================================
"""

import logging
import os

from config import CFG

log = logging.getLogger("deepshield")

# Paths come from config; these aliases keep the code below readable.
BASE_DIR = CFG.BASE_DIR
CKPT_PATH = CFG.CKPT_PATH
ONNX_PATH = CFG.ONNX_PATH
ONNX_META_PATH = CFG.ONNX_META_PATH
YUNET_PATH = CFG.YUNET_PATH

# The only place an architecture is turned into human-facing text. app.py
# reads these so no model name is ever written twice.
ARCH_NAMES = {
    "mobilenet_v3_small": "MobileNetV3-Small",
    "mobilenet_v3_large": "MobileNetV3-Large",
}
ARCH_PARAMS = {
    "mobilenet_v3_small": "2.5M",
    "mobilenet_v3_large": "5.4M",
}


def version_from(meta: dict) -> str:
    """Training run identifier, e.g. 'V3-Max'.

    Older checkpoints predate the explicit field, so it is recovered from
    the leading token of `trained_on` ("V3-Max multi-generator: …") rather
    than being invented here."""
    described = str(meta.get("trained_on") or "")
    first = described.split(":")[0].split()[0].strip() if described else ""
    return first if first.lower().startswith("v") else "unversioned"


def risk_for(prediction: str, confidence: int) -> str:
    """Risk label for a verdict — one definition, used by every caller."""
    if prediction == "deepfake":
        return "High" if confidence >= 85 else "Medium"
    return "Low" if confidence >= 80 else "Medium"


def certainty_for(confidence: int) -> str:
    """How strong the evidence is — not how probable the verdict is.

    The distinction matters. `confidence` is the winning class's softmax
    output, and softmax outputs are not probabilities unless the model has
    been calibrated, which this one has not. Saying "94% chance it is fake"
    claims a frequency nobody has measured. Saying "very strong evidence"
    claims a ranking, which is what the number actually supports.

    Bands live in config so the API, the UI and the evaluation harness all
    read the same table."""
    for lower, key, _label in CFG.CERTAINTY_BANDS:
        if confidence >= lower:
            return key
    return CFG.CERTAINTY_BANDS[-1][1]


def aggregate_frames(p_fakes, weights=None, topk_fraction=None,
                     suspicious_at=None) -> dict:
    """Turn per-frame P(fake) into one score, and show the working.

    Pure: a list of floats in, a dict out, no model and no I/O — which is
    why `scripts/video_test.py` can pin its behaviour against sequences
    whose right answer is obvious by construction.

    Three summaries, because each one is blind to a different thing:

        median  survives a handful of bad frames untouched
        mean    notices when the whole clip is slightly off
        top-k   finds manipulation confined to a few seconds, using k
                frames rather than one so a single outlier cannot decide

    Weights come from config and are provisional; every component is
    returned so the combination can be recomputed from the response alone.
    """
    import statistics

    ps = [float(p) for p in p_fakes]
    if not ps:
        raise ValueError("no frame scores to aggregate")

    w = dict(weights or CFG.VIDEO_WEIGHTS)
    allowed = {"median", "mean", "top_k"}
    unknown = set(w) - allowed
    if unknown:
        raise ValueError(
            "unknown video aggregation weight(s): " + ", ".join(sorted(unknown))
        )
    fraction = CFG.VIDEO_TOPK_FRACTION if topk_fraction is None else topk_fraction
    threshold = CFG.VIDEO_SUSPICIOUS_AT if suspicious_at is None else suspicious_at

    k = max(1, round(fraction * len(ps)))
    top = sorted(ps, reverse=True)[:k]

    parts = {"median": statistics.median(ps),
             "mean": sum(ps) / len(ps),
             "top_k": sum(top) / len(top)}

    total = sum(w.values()) or 1.0
    score = sum(parts[name] * weight for name, weight in w.items()) / total

    suspicious = [p for p in ps if p >= threshold]
    return {
        "score": score,
        "components": {n: round(v, 4) for n, v in parts.items()},
        "weights": dict(w),
        "k": k,
        "frames": len(ps),
        "suspicious": len(suspicious),
        "suspiciousAt": threshold,
        "peak": max(ps),
        "lowest": min(ps),
        "variance": statistics.pvariance(ps) if len(ps) > 1 else 0.0,
    }


def timestamp(seconds) -> str:
    """Seconds → "MM:SS", so a suspicious frame can be scrubbed to."""
    total = max(0, int(round(float(seconds))))
    return f"{total // 60:02d}:{total % 60:02d}"


def temporal_signals(records) -> dict:
    """Frame-to-frame consistency of the face itself.

    **Descriptive only. Nothing here votes on the verdict**, and that is a
    decision rather than an oversight: no labelled video set has been
    scored, so there is no evidence for what value of "landmark jitter"
    means manipulation. A signal nobody has validated must not be allowed
    to change an answer — it can only describe one.

    Cheap by design. Every number below comes from the face box, the five
    YuNet landmarks and a 32x32 thumbnail that were already computed to
    classify the frame, so this costs no extra forward passes and no extra
    detection. That is the whole reason it is not a video transformer.

    → each value is None when too few frames carried a face to measure it.
    """
    import numpy as np

    faces = [r for r in records if r.get("box")]
    out = {"facesFound": len(faces), "framesSampled": len(records)}

    if len(faces) < 2:
        return {**out, "facePositionJitter": None, "faceSizeJitter": None,
                "landmarkJitter": None, "appearanceContinuity": None}

    # Position: where the face sits, as a fraction of the frame. A steady
    # head gives a small number; a face that jumps between frames does not.
    cx = np.array([(r["box"][0] + r["box"][2] / 2) / r["frame"][0] for r in faces])
    cy = np.array([(r["box"][1] + r["box"][3] / 2) / r["frame"][1] for r in faces])
    out["facePositionJitter"] = round(float((cx.std() + cy.std()) / 2), 4)

    # Size: relative spread of the face's scale across the clip.
    size = np.array([np.sqrt(max(r["box"][2] * r["box"][3], 1e-6)) for r in faces])
    out["faceSizeJitter"] = round(float(size.std() / size.mean()), 4) if size.mean() else None

    # Landmarks: how far the five points move between consecutive sampled
    # frames, in units of face width — so it does not grow just because the
    # subject walked towards the camera.
    steps = []
    for a, b in zip(faces, faces[1:]):
        if not (a.get("landmarks") and b.get("landmarks")):
            continue
        width = max(b["box"][2], 1e-6)
        shared = set(a["landmarks"]) & set(b["landmarks"])
        if not shared:
            continue
        moved = []
        for name in shared:
            ax, ay = a["landmarks"][name]; ax += a["origin"][0]; ay += a["origin"][1]
            bx, by = b["landmarks"][name]; bx += b["origin"][0]; by += b["origin"][1]
            moved.append(np.hypot(bx - ax, by - ay) / width)
        steps.append(float(np.mean(moved)))
    out["landmarkJitter"] = round(float(np.mean(steps)), 4) if steps else None

    # Appearance: correlation between consecutive face thumbnails. A real
    # face changes smoothly; this is the cheapest stand-in for the
    # frame-to-frame embedding similarity a heavier model would compute,
    # and it is named for what it measures rather than what it imitates.
    sims = []
    for a, b in zip(faces, faces[1:]):
        ta, tb = a.get("thumb"), b.get("thumb")
        if ta is None or tb is None:
            continue
        va, vb = np.asarray(ta, float).ravel(), np.asarray(tb, float).ravel()
        if va.std() < 1e-6 or vb.std() < 1e-6:
            continue
        sims.append(float(np.corrcoef(va, vb)[0, 1]))
    out["appearanceContinuity"] = round(float(np.mean(sims)), 4) if sims else None

    return out


def certainty_bands() -> list:
    """The band table, for anything that has to label a number it was
    given — so no threshold is ever written down twice."""
    return [{"from": lower, "to": (CFG.CERTAINTY_BANDS[i - 1][0] if i else 100),
             "key": key, "label": label}
            for i, (lower, key, label) in enumerate(CFG.CERTAINTY_BANDS)]

# ---- Ensemble: pretrained HuggingFace verifiers (images only) ----
# OPT-IN (DS_VERIFIERS=1). They mattered when our model was blind to
# StyleGAN2, but V3 covers those fakes itself and scores every image in
# our held set correctly alone — while the verifiers cost ~1 GB of disk
# and RAM and have flagged authentic photos (SigLIP scored a re-saved
# real portrait 1.00). Turn them on to cross-check a verdict.
HF_MODELS = [
    {"id": "prithivMLmods/Deep-Fake-Detector-v2-Model", "name": "ViT Deepfake v2"},
    {"id": "Ateeqq/ai-vs-human-image-detector",         "name": "SigLIP AI-image"},
]


def verifiers_enabled() -> bool:
    return CFG.VERIFIERS
import re as _re
_FAKE_LABEL = _re.compile(r"fake|deep|ai|artificial|synthetic|generat", _re.I)

_engine = None  # lazy singleton


# ---------------------------------------------------------- availability

def torch_available() -> bool:
    try:
        import torch  # noqa: F401
        import torchvision  # noqa: F401
        return True
    except Exception:
        log.debug("torch/torchvision unavailable", exc_info=True)
        return False


def onnx_available() -> bool:
    """The lean path: OpenCV runs the network, so PyTorch is not needed.
    Requires the exported pair written by scripts/export_onnx.py."""
    return os.path.exists(ONNX_PATH) and os.path.exists(ONNX_META_PATH)


def engine_available() -> bool:
    """True when a runnable model exists — ONNX (preferred) or a PyTorch
    checkpoint. DS_ENGINE=echo forces the openly-labeled simulated mode
    (the UI then shows the yellow 'Simulated (demo)' badge)."""
    if CFG.FORCE_ECHO:
        return False
    if onnx_available():
        return True
    return os.path.exists(CKPT_PATH) and torch_available()


def engine_info() -> dict:
    """Metadata for /api/health (accuracy comes from the checkpoint)."""
    if not engine_available():
        return {}
    return {**_get_engine().info, "verifiers": verifiers_enabled()}


# ---------------------------------------------------------- engine

class _Engine:
    """Runs the trained classifier on CPU.

    Two interchangeable backends, both fed by the same preprocessing:
      'onnx'  — OpenCV's DNN module runs the exported graph. Default,
                and the reason the deployment needs no PyTorch at all.
      'torch' — the original .pth path, used when no ONNX export exists.
    Verified to agree to ~1e-7 on every probability, so which one is
    active never changes a verdict.
    """

    def __init__(self):
        import numpy as np
        self.np = np

        if onnx_available():
            self._init_onnx()
        else:
            self._init_torch()

        self.info = {
            "engine": "live",
            # ---- identity: one block, read from the model's own metadata ----
            "model_name": self.meta.get("model_name", "DeepShield"),
            "architecture": ARCH_NAMES.get(self.arch, self.arch),
            "version": self.meta.get("version") or version_from(self.meta),
            "runtime": "ONNX" if self.backend == "onnx" else "PyTorch",
            "input_size": self.size,
            "classes": list(self.classes),
            # ---- provenance ----
            "backend": self.backend,
            "checkpoint": self.checkpoint_name,
            "arch": self.arch,
            "params": ARCH_PARAMS.get(self.arch),
            # ---- measured performance ----
            "val_accuracy": self.meta.get("val_accuracy"),
            "test_accuracy": self.meta.get("test_accuracy"),
            "tpdn_accuracy": self.meta.get("tpdn_accuracy"),
            "dfdc_accuracy": self.meta.get("dfdc_accuracy"),
            "trained_on": self.meta.get("trained_on"),
        }

        self._detector = None  # lazy YuNet face detector (OpenCV 5 DNN)

    # ---- backend: ONNX through OpenCV (no torch anywhere) ----
    def _init_onnx(self):
        import cv2
        import json

        with open(ONNX_META_PATH, encoding="utf-8") as f:
            self.meta = json.load(f)
        self.backend = "onnx"
        self.checkpoint_name = os.path.basename(ONNX_PATH)
        self.arch = self.meta.get("arch", "mobilenet_v3_large")
        self.classes = self.meta["classes"]      # ['fake', 'real'] — order matters
        self.size = self.meta.get("input_size", 224)
        self.norm = self.meta["normalize"]
        self.net = cv2.dnn.readNetFromONNX(ONNX_PATH)
        self.torch = None

    # ---- backend: the original PyTorch checkpoint ----
    def _init_torch(self):
        import torch
        from torchvision import models

        ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=True)
        self.meta = ckpt
        self.backend = "torch"
        self.checkpoint_name = os.path.basename(CKPT_PATH)
        self.classes = ckpt["classes"]
        self.size = ckpt.get("input_size", 224)
        self.norm = ckpt.get("normalize", {"mean": [0.485, 0.456, 0.406],
                                           "std":  [0.229, 0.224, 0.225]})

        arch = ckpt.get("arch", "mobilenet_v3_small")
        builders = {
            "mobilenet_v3_small": models.mobilenet_v3_small,
            "mobilenet_v3_large": models.mobilenet_v3_large,
        }
        if arch not in builders:
            raise ValueError(f"Unsupported arch in checkpoint: {arch}")
        self.arch = arch

        model = builders[arch](weights=None)
        model.classifier[3] = torch.nn.Linear(
            model.classifier[3].in_features, len(self.classes))
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))  # keep the i3 responsive
        self.model = model
        self.torch = torch

    # ---- shared preprocessing: PIL resize + normalise, in numpy ----
    # Matches torchvision's Resize→ToTensor→Normalize exactly (torchvision
    # resizes PIL images with PIL itself), so both backends see the same
    # array and the swap stays invisible in the results.
    def _to_input(self, pil_image):
        from PIL import Image
        np = self.np
        img = pil_image.convert("RGB").resize((self.size, self.size), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0          # HWC, 0-1
        arr = (arr - np.array(self.norm["mean"], dtype=np.float32)) \
            / np.array(self.norm["std"], dtype=np.float32)
        return arr.transpose(2, 0, 1)                            # CHW

    # ---- shared forward: (N,3,H,W) float32 → (N,classes) probabilities ----
    # OpenCV's DNN engine dies on large batches (36 crashed the process,
    # 16 is fine), so requests are chunked rather than trusted whole.
    MAX_BATCH = CFG.MAX_FORWARD_BATCH

    def _forward(self, batch):
        np = self.np
        parts = []
        for i in range(0, len(batch), self.MAX_BATCH):
            chunk = np.ascontiguousarray(batch[i:i + self.MAX_BATCH], dtype=np.float32)
            if self.backend == "onnx":
                self.net.setInput(chunk)
                parts.append(self.net.forward())
            else:
                with self.torch.no_grad():
                    parts.append(self.model(self.torch.from_numpy(chunk)).numpy())
        logits = np.concatenate(parts, axis=0)
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

        self._detector = None  # lazy YuNet face detector (OpenCV 5 DNN)

    # ---- face crop: align inference with the training domain ----
    # The dataset is tight face portraits; feeding whole photos
    # (background, clothes, scenery) biases the model toward "real".
    def _face_crop(self, pil_image):
        crop, _ = self._face_crop_ex(pil_image)
        return crop

    def _face_crop_ex(self, pil_image):
        """(crop, landmarks) — the long-standing shape, used by explain()."""
        found = self._detect_face(pil_image)
        return found["crop"], found["landmarks"]

    def _detect_face(self, pil_image):
        """The largest face, as everything before multi-face support saw it.

        The video path and the temporal signals are written around one face
        per frame, so they keep this."""
        found = self._detect_faces(pil_image, limit=1)
        return found[0]

    def _detect_faces(self, pil_image, limit=None):
        """Every face, largest first. Always at least one entry.

        → [{crop, landmarks, box, origin, frame, found}, ...]

        This used to return only the largest, which is wrong for the single
        most common real deepfake: a group photograph with one swapped face.
        A real portrait beside a swapped one outvoted it by being a few
        pixels wider, and the app answered "real 97%" with a fake face
        plainly in the frame.

        `box` is (x, y, w, h) of the face and `origin` the crop's top-left
        corner, both in the capped frame's coordinates. The video path
        needs them to compare where a face sits from frame to frame;
        `_face_crop_ex` throws them away and behaves exactly as before.
        """
        import cv2
        import numpy as np
        from PIL import Image

        miss = {"crop": pil_image, "landmarks": None, "box": None,
                "origin": (0, 0), "frame": pil_image.size, "found": False}

        if not os.path.exists(YUNET_PATH):
            return [miss]  # detector missing → analyze full frame

        if self._detector is None:
            self._detector = cv2.FaceDetectorYN_create(
                YUNET_PATH, "", (320, 320), 0.6, 0.3, 5000)

        rgb = np.array(pil_image.convert("RGB"))

        # Cap the working resolution. Cropping straight out of a very large
        # photo hands the model a downsampling path it never saw in training
        # (dataset faces are ~256px): a 2687px press portrait scored 0.94
        # fake, the same photo at 1024px scored 0.02. Normalising the scale
        # first removes that artefact.
        MAX_SIDE = CFG.MAX_IMAGE_SIDE
        if max(rgb.shape[:2]) > MAX_SIDE:
            s = MAX_SIDE / max(rgb.shape[:2])
            rgb = cv2.resize(rgb, (int(rgb.shape[1] * s), int(rgb.shape[0] * s)),
                             interpolation=cv2.INTER_AREA)
        h, w = rgb.shape[:2]

        # Detection runs on the (already capped) image
        scale = 1.0
        det_img = rgb

        bgr = cv2.cvtColor(det_img, cv2.COLOR_RGB2BGR)
        self._detector.setInputSize((bgr.shape[1], bgr.shape[0]))
        _, faces = self._detector.detect(bgr)
        if faces is None or len(faces) == 0:
            return [{**miss, "frame": (w, h)}]  # no face → whole frame

        # Largest first: the main subject still leads, and the cap below
        # drops the smallest faces rather than the ones anyone is looking at.
        order = np.argsort(-(faces[:, 2] * faces[:, 3]))
        cap = CFG.MAX_FACES if limit is None else limit
        names = ["right_eye", "left_eye", "nose", "mouth_right", "mouth_left"]

        out = []
        for index in order[:max(1, cap)]:
            best = faces[int(index)]
            x, y, fw, fh = [v / scale for v in best[:4]]

            m = 0.35 * max(fw, fh)  # margin, matches portrait-style crops
            x0 = max(0, int(x - m));  y0 = max(0, int(y - m))
            x1 = min(w, int(x + fw + m));  y1 = min(h, int(y + fh + m))
            if x1 <= x0 or y1 <= y0:
                continue

            # YuNet landmarks (5 points) → crop coordinates, for the
            # explainability "focus region" text
            landmarks = {}
            for i, name in enumerate(names):
                lx = best[4 + i * 2] / scale - x0
                ly = best[5 + i * 2] / scale - y0
                landmarks[name] = (float(lx), float(ly))

            out.append({"crop": Image.fromarray(rgb[y0:y1, x0:x1]),
                        "landmarks": landmarks,
                        "box": (float(x), float(y), float(fw), float(fh)),
                        "origin": (int(x0), int(y0)),
                        "frame": (w, h), "found": True})

        return out or [{**miss, "frame": (w, h)}]

    # ---- probability vector over classes (input already face-cropped)
    # TTA: averages the prediction over the image + its mirror — a small
    # free accuracy/stability boost for ~2x compute (still <0.2s on CPU).
    def _probs_raw(self, pil_image):
        img = self._normalize_compression(pil_image.convert("RGB"))
        batch = self.np.stack([
            self._to_input(img),
            self._to_input(img.transpose(0)),   # PIL FLIP_LEFT_RIGHT
        ])
        return self._forward(batch).mean(axis=0)

    @staticmethod
    def _normalize_compression(img, quality=CFG.JPEG_NORMALISE_QUALITY):
        """Put every input in the same compression domain the model trained
        on. Training saw JPEG-recompressed faces (q30-95); a pristine
        camera original carries high-frequency detail it never learned as
        'normal' and scored 0.95 fake, while the same photo re-saved as
        JPEG scored 0.02. One round-trip removes that mismatch.
        Applied to our model only — the verifiers do their own thing, and
        SigLIP in particular reacts badly to re-encoding."""
        import io
        from PIL import Image
        try:
            buf = io.BytesIO()
            img.save(buf, "JPEG", quality=quality)
            buf.seek(0)
            return Image.open(buf).convert("RGB")
        except Exception:
            return img

    # ---- crop + classify (used by the video path per-frame)
    def _probs(self, pil_image):
        return self._probs_raw(self._face_crop(pil_image))

    def predict_image(self, image_path):
        """→ ((prediction 'real'|'deepfake', confidence int), frames=1)"""
        from PIL import Image
        # Context manager releases the file handle — without it Windows
        # blocks the post-analysis delete of the uploaded file.
        with Image.open(image_path) as im:
            probs = self._probs(im)
        return self._verdict(probs), 1

    def predict_video(self, video_path, frame_rate=CFG.DEFAULT_FRAME_RATE, max_frames=CFG.MAX_VIDEO_FRAMES):
        """Sample ~frame_rate frames/sec (CPU-friendly) and score each one.

        → (records, meta). Every frame keeps its own P(fake), its timestamp
        and the face geometry the temporal signals need. Combining those
        into a verdict is `aggregate_frames`' job, deliberately outside
        this method: extraction needs a video file, aggregation does not,
        and only one of the two can then be tested without one.
        """
        import cv2
        from PIL import Image

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Could not open video")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps != fps or fps <= 0:   # 0, negative or NaN header
            fps = 25.0
        step = max(1, round(fps / max(0.25, frame_rate)))  # every Nth frame
        fake_index = self.classes.index("fake")

        records, idx = [], 0
        try:
            while len(records) < max_frames:
                # Sampling at 1 fps from a 30 fps clip discards 29 frames out of
                # every 30. `read()` fully decodes and colour-converts each one
                # first; `grab()` advances the stream without paying for a frame
                # nobody looks at. Same frames chosen, same scores, less work.
                if idx % step:
                    if not cap.grab():
                        break
                    idx += 1
                    continue

                ok, frame = cap.read()
                if not ok:
                    break

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                found = self._detect_face(Image.fromarray(rgb))
                probs = self._probs_raw(found["crop"])

                # 32x32 grey thumbnail of the crop we already made - the only
                # extra work the temporal signals add per frame.
                grey = cv2.cvtColor(self.np.array(found["crop"].convert("RGB")),
                                    cv2.COLOR_RGB2GRAY)
                thumb = cv2.resize(grey, (32, 32), interpolation=cv2.INTER_AREA)

                records.append({
                    "index": idx,
                    "time": idx / fps,
                    "pFake": float(probs[fake_index]),
                    "box": found["box"],
                    "origin": found["origin"],
                    "frame": found["frame"],
                    "landmarks": found["landmarks"],
                    "thumb": thumb,
                })
                idx += 1
        finally:
            cap.release()

        if not records:
            raise ValueError("No readable frames in video")
        return records, {"fps": float(fps), "step": int(step),
                         "duration": idx / fps}

    # ---- Explainability: occlusion sensitivity ----
    # Blank out one patch at a time and watch the verdict move: the
    # regions whose removal changes the score the most are the ones the
    # model was relying on. Forward passes only, so it behaves identically
    # on both backends (Grad-CAM would need gradients, which the ONNX
    # runtime cannot provide) — and it is easier to justify: we measure
    # the model's dependence rather than interpret its internals.
    def explain(self, face_img, landmarks, grid=CFG.OCCLUSION_GRID):
        import cv2
        import base64
        np = self.np

        img = self._normalize_compression(face_img.convert("RGB"))
        base_input = self._to_input(img)
        # One forward, read twice — this used to run the identical tensor
        # through the network a second time to fetch a value it already had.
        base_probs = self._forward(base_input[None])
        cls = int(base_probs.argmax())
        base_score = float(base_probs[0, cls])

        # One batch: the image with each patch greyed out in turn
        S = self.size
        patch = S // grid
        variants, cells = [], []
        for gy in range(grid):
            for gx in range(grid):
                v = base_input.copy()
                y0, x0 = gy * patch, gx * patch
                v[:, y0:y0 + patch, x0:x0 + patch] = 0.0   # 0 = dataset mean
                variants.append(v)
                cells.append((gy, gx))

        scores = self._forward(np.stack(variants))[:, cls]
        cam = np.zeros((grid, grid), dtype=np.float32)
        for (gy, gx), s in zip(cells, scores):
            cam[gy, gx] = max(0.0, base_score - float(s))  # drop = reliance

        if cam.max() > 0:
            cam = cam / cam.max()

        # Hotspot in face-crop pixel coordinates
        iy, ix = np.unravel_index(int(np.argmax(cam)), cam.shape)
        fw, fh_ = face_img.size
        hot_x = (ix + 0.5) / cam.shape[1] * fw
        hot_y = (iy + 0.5) / cam.shape[0] * fh_

        # Focus-region text, grounded in YuNet landmarks (no fabrication)
        REGION_OF = {
            "right_eye": "the eye region", "left_eye": "the eye region",
            "nose": "the nose area",
            "mouth_right": "the mouth area", "mouth_left": "the mouth area",
        }

        def region_at(px, py):
            """Nearest landmark, named. No landmarks means no claim."""
            if not landmarks:
                return None
            nearest = min(landmarks.items(),
                          key=lambda kv: (kv[1][0] - px) ** 2 + (kv[1][1] - py) ** 2)[0]
            return REGION_OF[nearest]

        region = region_at(hot_x, hot_y) or "the central face region"

        # Every region the prediction leaned on, not just the top one. The
        # grid was already scored, so this costs nothing — and one region is
        # a poor summary when a face-swap gives itself away at both the eyes
        # and the mouth. Each weight is the largest normalised drop any cell
        # in that region produced.
        regions = {}
        for (gy, gx), value in zip(cells, cam.reshape(-1)):
            if value <= 0:
                continue
            px = (gx + 0.5) / grid * fw
            py = (gy + 0.5) / grid * fh_
            name = region_at(px, py)
            if name:
                regions[name] = max(regions.get(name, 0.0), float(value))

        ranked = sorted(regions.items(), key=lambda kv: kv[1], reverse=True)
        # Drop the also-rans: a region that barely moved the score is noise
        # dressed up as an explanation.
        top = [{"name": n, "weight": round(w, 3)} for n, w in ranked
               if w >= 0.25 * ranked[0][1]][:3] if ranked else []

        # Heatmap overlay image (JPEG data URL)
        base = cv2.resize(np.array(face_img.convert("RGB")), (224, 224))
        heat = cv2.resize((cam * 255).astype(np.uint8), (224, 224),
                          interpolation=cv2.INTER_CUBIC)  # smooth the coarse grid
        heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(cv2.cvtColor(base, cv2.COLOR_RGB2BGR), 0.55,
                                  heat, 0.45, 0)
        ok, buf = cv2.imencode(".jpg", overlay, [cv2.IMWRITE_JPEG_QUALITY, 82])
        data_url = ("data:image/jpeg;base64," +
                    base64.b64encode(buf).decode()) if ok else None

        return {
            "heatmapDataUrl": data_url,
            "focusRegion": region,
            "regions": top,
            "method": "occlusion sensitivity",
            "note": f"Prediction was most sensitive to {region}.",
        }

    # ---- probs → (prediction, confidence)
    def _verdict(self, probs):
        top = int(probs.argmax())
        label = self.classes[top]                     # 'fake' or 'real'
        prediction = "deepfake" if label == "fake" else "real"
        confidence = int(round(float(probs[top]) * 100))
        return prediction, confidence


_engine_mtime = None


def _active_model_path() -> str:
    """Whichever file the engine will actually load."""
    return ONNX_PATH if onnx_available() else CKPT_PATH


def _get_engine() -> _Engine:
    """Singleton, but reloads automatically if the model file is replaced
    (dropping in a newly trained model needs no restart)."""
    global _engine, _engine_mtime
    path = _active_model_path()
    try:
        stamp = (path, os.path.getmtime(path))
    except OSError:
        if _engine is not None:
            return _engine
        raise
    if _engine is None or _engine_mtime != stamp:
        _engine = _Engine()
        _engine_mtime = stamp
    return _engine


# ---------------------------------------------------------- HF verifiers

class _HFEngine:
    """Generic wrapper around a HuggingFace image-classification model.
    Maps whatever labels the model uses onto P(fake) via regex."""

    def __init__(self, model_id, name):
        import torch
        from transformers import AutoImageProcessor, AutoModelForImageClassification

        self.name = name
        self.torch = torch
        # local_files_only: inference must NEVER wait on the network —
        # a partially-downloaded model fails instantly and gets skipped.
        self.processor = AutoImageProcessor.from_pretrained(model_id, local_files_only=True)
        self.model = AutoModelForImageClassification.from_pretrained(model_id, local_files_only=True)
        self.model.eval()

        # Which output index means "fake"?
        self.fake_idx = None
        for idx, label in self.model.config.id2label.items():
            if _FAKE_LABEL.search(str(label)):
                self.fake_idx = int(idx)
                break
        if self.fake_idx is None:
            raise ValueError(f"{model_id}: no fake-like label in {self.model.config.id2label}")

    def p_fake(self, pil_image):
        inputs = self.processor(images=pil_image.convert("RGB"), return_tensors="pt")
        with self.torch.no_grad():
            probs = self.torch.softmax(self.model(**inputs).logits, dim=1)[0]
        return float(probs[self.fake_idx])


_hf_engines = None  # lazy: None = not tried yet, [] = tried, none loaded


def _get_hf_engines():
    global _hf_engines
    if not verifiers_enabled():
        return []
    if _hf_engines is None:
        _hf_engines = []
        for cfg in HF_MODELS:
            try:
                _hf_engines.append(_HFEngine(cfg["id"], cfg["name"]))
            except Exception:
                pass  # offline / not downloaded / transformers missing → skip
    return _hf_engines


# ---------------------------------------------------------- public API

def score_image(path) -> float:
    """→ P(fake) for one image, through the exact preprocessing a real
    request gets: face crop, compression normalisation, flip-averaged.

    Evaluation needs the probability rather than the rounded verdict, and
    it must not pay for the occlusion heatmap — 36 extra forward passes
    per image would put a 10,000-image benchmark out of reach. Scoring a
    different way than serving would measure a model nobody uses, so this
    shares `_probs` with the request path instead of reimplementing it."""
    from PIL import Image
    eng = _get_engine()
    with Image.open(path) as im:
        probs = eng._probs(im)
    return float(probs[eng.classes.index("fake")])


def analyze_file(path, file_type, frame_rate=CFG.DEFAULT_FRAME_RATE):
    """Main entry for app.py.
    → dict {prediction, confidence, framesAnalyzed, ensemble?}"""
    eng = _get_engine()

    # Video: our fast model only — running ViTs per-frame is too slow on CPU
    if file_type == "video":
        records, meta = eng.predict_video(path, frame_rate)
        agg = aggregate_frames([r["pFake"] for r in records])
        score = agg["score"]
        prediction = "deepfake" if score >= 0.5 else "real"
        confidence = int(round((score if score >= 0.5 else 1 - score) * 100))

        hottest = sorted(records, key=lambda r: r["pFake"], reverse=True)
        return {
            "prediction": prediction,
            "confidence": confidence,
            "framesAnalyzed": len(records),
            # True when any sampled frame carried a face. `video.temporal`
            # has the per-frame counts; this is the one bit the result page
            # needs to decide whether to warn.
            "faceFound": any(r.get("box") for r in records),
            "video": {
                "framesAnalyzed": len(records),
                "suspiciousFrames": agg["suspicious"],
                "suspiciousAt": agg["suspiciousAt"],
                "peakFakeScore": round(agg["peak"], 4),
                "medianFakeScore": round(agg["components"]["median"], 4),
                "meanFakeScore": round(agg["components"]["mean"], 4),
                "topKFakeScore": round(agg["components"]["top_k"], 4),
                "lowestFakeScore": round(agg["lowest"], 4),
                "scoreVariance": round(agg["variance"], 6),
                "combinedScore": round(score, 4),
                "k": agg["k"],
                "weights": agg["weights"],
                "topTimestamps": [
                    {"time": round(r["time"], 2), "timestamp": timestamp(r["time"]),
                     "score": round(r["pFake"], 4)}
                    for r in hottest[:CFG.VIDEO_TOP_TIMESTAMPS]],
                "timeline": [{"t": round(r["time"], 2), "p": round(r["pFake"], 4)}
                             for r in records],
                "temporal": temporal_signals(records),
                "fps": round(meta["fps"], 2),
                "sampledEveryNthFrame": meta["step"],
                "durationSeconds": round(meta["duration"], 2),
            },
            # No longer None: the frame scores do combine into one number,
            # and hiding it left the result page with nothing to show.
            "ensemble": [{"model": "MobileNetV3 (ours)", "pFake": round(score, 4),
                          "weight": 1,
                          "note": "video — median / mean / top-k over frames"}],
        }

    # Image: ensemble — our model + every available HF verifier vote P(fake)
    from PIL import Image
    with Image.open(path) as im:
        # Every face, not the largest one. `_face_crop_ex` also threw away
        # whether a face was found at all, which the caller has to be told:
        # analysing a whole frame is a reasonable fallback, presenting the
        # result as though a face were in it is not.
        detected = eng._detect_faces(im)
        fake_i = eng.classes.index("fake")

        # One image, one verdict — so the faces have to be reduced to one
        # score, and the max is the only defensible choice. A photograph
        # containing a manipulated face is a manipulated photograph, whatever
        # the other people in it look like. Averaging would let a crowd
        # outvote the swap, which is exactly the attack.
        scored = [(eng._probs_raw(d["crop"]), d) for d in detected]
        probs, found = max(scored, key=lambda pair: float(pair[0][fake_i]))

        face, landmarks = found["crop"], found["landmarks"]
        votes = [{"model": "MobileNetV3 (ours)", "pFake": round(float(probs[fake_i]), 4)}]

        for hf in _get_hf_engines():
            try:
                votes.append({"model": hf.name, "pFake": round(hf.p_fake(face), 4)})
            except Exception:
                pass

        # Explainability (occlusion-sensitivity heatmap + grounded text) —
        # best-effort: any failure just omits the section
        try:
            explain = eng.explain(face, landmarks)
        except Exception:
            explain = None

    # ---- Combining the votes ----
    # Our own model leads. Noisy-OR made sense while it was blind to
    # StyleGAN2, but that blindness is gone (V3 trains on those fakes) and
    # trusting any confident vote backfires: SigLIP scores a re-saved
    # authentic photo at 1.00 — it recognises processing, not manipulation.
    # A false "fake" on someone's real photo is the worst outcome we can
    # produce, so verifiers now advise rather than decide.
    OWN_WEIGHT = CFG.OWN_WEIGHT
    if len(votes) > 1:
        w_hf = (1.0 - OWN_WEIGHT) / (len(votes) - 1)
        weights = [OWN_WEIGHT] + [w_hf] * (len(votes) - 1)
    else:
        weights = [1.0]
    for v, w in zip(votes, weights):
        v["weight"] = round(w, 3)

    p = sum(v["pFake"] * v["weight"] for v in votes)

    # Verifiers can still overrule, but only together and only when both
    # are near-certain — one biased verifier is never enough.
    verifiers = [v["pFake"] for v in votes[1:]]
    if verifiers and all(x >= CFG.VERIFIER_OVERRULE_AT for x in verifiers):
        p = max(p, sum(verifiers) / len(verifiers))

    prediction = "deepfake" if p >= 0.5 else "real"
    confidence = int(round((p if p >= 0.5 else 1 - p) * 100))
    disputed = any((v["pFake"] >= 0.5) != (p >= 0.5) for v in votes)

    return {
        "prediction": prediction,
        "confidence": confidence,
        "framesAnalyzed": 1,
        # False means no face was found and the whole frame was scored. The
        # model was trained on faces only, so a verdict on a landscape or a
        # screenshot is confident and meaningless - and until now it looked
        # exactly like a verdict on a face.
        "faceFound": bool(found["found"]),
        # How many were found, and where in that ranking the reported verdict
        # came from. On a group photo the result page has to be able to say
        # "one of four faces" rather than implying the whole picture.
        "facesFound": len([d for d in detected if d["found"]]),
        "ensemble": votes,
        "disputed": disputed,
        "combiner": "own-led + verifier consensus",
        "explain": explain,
    }
