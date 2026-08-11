"""Production-equivalent preprocessing, for training and evaluation.

The dataset audit found that training and production preprocess differently,
which means every validation figure V3 produced describes a model that is not
the one being served. This module is the fix: one implementation of the
production path that a Kaggle notebook, a local test and the benchmark can
all import.

---------------------------------------------------------------- the contract

Production (`backend/inference.py`) does exactly this, in this order:

    1. PIL image -> RGB -> numpy
    2. if max(h, w) > 1024:  cv2.resize(..., INTER_AREA)   [truncating ints]
    3. RGB -> BGR for the detector
    4. YuNet: input (320, 320), score 0.6, nms 0.3, top_k 5000
    5. largest face by w*h
    6. margin = 0.35 * max(fw, fh), box clamped to the frame [truncating ints]
    7. crop taken from the RGB array
    8. JPEG round trip at quality 88
    9. resize to 224 x 224, PIL BILINEAR
   10. /255, subtract ImageNet mean, divide by ImageNet std, HWC -> CHW

Every constant above lives in `ProductionConfig` and nowhere else.

------------------------------------------------------- why this is a copy

This module deliberately does **not** import `backend/inference.py`. It has
to run inside a Kaggle session where the repository does not exist, so it
must stand on cv2, PIL and numpy alone.

A copy can drift, so the copy is pinned: `tests/test_preprocess_parity.py`
asserts this module and the production engine produce **bit-identical**
tensors on the sample images. If someone changes one and not the other, that
test fails. The duplication is deliberate; the drift is not permitted.

------------------------------------------------------- where augmentation goes

Production's last image-domain operation is the q88 round trip. If
augmentation ran after it, q88 would no longer be the final compression step
and the parity would be broken in exactly the way this module exists to
prevent. So augmentation is inserted **between the crop and the q88 tail**:

    crop  ->  [training only: augmentation]  ->  q88 -> resize -> normalise
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
              the only thing evaluation leaves out

`baseline_tensor()` is the deterministic path and is what validation,
benchmark evaluation and the sealed test set must use. `training_tensor()`
is the same path with the augmentation layer inserted. There is no third
path.

--------------------------------------------------------------- required file

YuNet is **not downloaded by this module**. It needs:

    models/face_detection_yunet.onnx        (232 KB)

which is already committed in this repository. For a Kaggle run, upload that
one file as a dataset input and pass its path to `Preprocessor(...)`. Its
upstream home is the OpenCV Zoo:

    https://github.com/opencv/opencv_zoo
    models/face_detection_yunet/face_detection_yunet_2023mar.onnx

If the file is absent, `Preprocessor` raises with that message rather than
falling back to something that would silently produce a different crop.
"""
from dataclasses import dataclass, field
import io
import os

import numpy as np

__all__ = ["ProductionConfig", "PRODUCTION", "Preprocessor",
           "FaceResult", "TrainingAugmentation"]


# ------------------------------------------------------------------ config

@dataclass(frozen=True)
class ProductionConfig:
    """Every constant the production path uses.

    Mirrors `backend/config.py`. Kept as data rather than literals so a
    test can compare the two field by field."""

    max_side: int = 1024                      # CFG.MAX_IMAGE_SIDE
    detector_input: tuple = (320, 320)        # YuNet construction size
    score_threshold: float = 0.6
    nms_threshold: float = 0.3
    top_k: int = 5000
    margin: float = 0.35                      # of max(face_w, face_h)
    jpeg_quality: int = 88                    # CFG.JPEG_NORMALISE_QUALITY
    input_size: int = 224
    mean: tuple = (0.485, 0.456, 0.406)
    std: tuple = (0.229, 0.224, 0.225)
    classes: tuple = ("fake", "real")         # index 0 is fake; order matters


PRODUCTION = ProductionConfig()


@dataclass
class FaceResult:
    """What one detection pass produced.

    `found=False` means no face: production analyses the whole frame in that
    case, and this mirrors it so the two agree. **Dataset preparation should
    drop these rather than train on them** — a frame with no face teaches the
    model to classify backgrounds — but that is the caller's decision, and
    the flag is what makes it possible."""

    crop: object                  # PIL.Image
    found: bool
    box: tuple = None             # (x, y, w, h) in capped-frame coordinates
    origin: tuple = (0, 0)        # crop top-left, capped-frame coordinates
    frame: tuple = None           # (w, h) of the capped frame
    landmarks: dict = field(default_factory=dict)
    score: float = None           # detector confidence, for quality filtering
    n_faces: int = 0


# ------------------------------------------------------------ preprocessor

class Preprocessor:
    """The production path, reimplemented and pinned by a parity test."""

    LANDMARK_NAMES = ("right_eye", "left_eye", "nose", "mouth_right", "mouth_left")

    def __init__(self, yunet_path, config=PRODUCTION):
        self.cfg = config
        self.yunet_path = str(yunet_path)
        if not os.path.exists(self.yunet_path):
            raise FileNotFoundError(
                f"YuNet model not found at {self.yunet_path}.\n"
                "This module does not download it. Use the copy committed at "
                "models/face_detection_yunet.onnx, or fetch "
                "face_detection_yunet_2023mar.onnx from "
                "https://github.com/opencv/opencv_zoo and pass its path.")
        self._detector = None

    # ---- step 1-2: colour and the resolution cap

    def cap_resolution(self, pil_image):
        """RGB numpy array, capped at max_side with INTER_AREA.

        The truncating `int()` calls are production's, not a rounding
        choice — matching them is the difference between an identical crop
        and an off-by-one one."""
        import cv2
        rgb = np.array(pil_image.convert("RGB"))
        longest = max(rgb.shape[:2])
        if longest > self.cfg.max_side:
            s = self.cfg.max_side / longest
            rgb = cv2.resize(rgb,
                             (int(rgb.shape[1] * s), int(rgb.shape[0] * s)),
                             interpolation=cv2.INTER_AREA)
        return rgb

    # ---- step 3-7: detection and crop

    def detect_face(self, pil_image) -> FaceResult:
        import cv2
        from PIL import Image

        rgb = self.cap_resolution(pil_image)
        h, w = rgb.shape[:2]
        miss = FaceResult(crop=Image.fromarray(rgb), found=False, frame=(w, h))

        if self._detector is None:
            self._detector = cv2.FaceDetectorYN_create(
                self.yunet_path, "", self.cfg.detector_input,
                self.cfg.score_threshold, self.cfg.nms_threshold, self.cfg.top_k)

        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        self._detector.setInputSize((bgr.shape[1], bgr.shape[0]))
        _, faces = self._detector.detect(bgr)
        if faces is None or len(faces) == 0:
            return miss

        best = faces[int(np.argmax(faces[:, 2] * faces[:, 3]))]
        x, y, fw, fh = [float(v) for v in best[:4]]

        m = self.cfg.margin * max(fw, fh)
        x0 = max(0, int(x - m))
        y0 = max(0, int(y - m))
        x1 = min(w, int(x + fw + m))
        y1 = min(h, int(y + fh + m))
        if x1 <= x0 or y1 <= y0:
            return miss

        landmarks = {}
        for i, name in enumerate(self.LANDMARK_NAMES):
            landmarks[name] = (float(best[4 + i * 2]) - x0,
                               float(best[5 + i * 2]) - y0)

        return FaceResult(
            crop=Image.fromarray(rgb[y0:y1, x0:x1]),
            found=True,
            box=(x, y, fw, fh),
            origin=(int(x0), int(y0)),
            frame=(w, h),
            landmarks=landmarks,
            score=float(best[-1]) if len(best) > 14 else None,
            n_faces=int(len(faces)),
        )

    def crop(self, pil_image):
        """Just the crop — what dataset preparation saves to disk.

        Save it losslessly. A JPEG intermediate bakes one compression
        generation into every sample, which is precisely the shortcut V1
        learned instead of learning generator artefacts."""
        return self.detect_face(pil_image).crop

    # ---- step 8: the compression domain

    def normalize_compression(self, pil_image):
        """One JPEG round trip at q88, unconditionally.

        Production applies this to every input. A pristine camera original
        carries high-frequency detail the model never saw as normal and
        scored 0.95 fake; the same photo re-saved as JPEG scored 0.02."""
        from PIL import Image
        try:
            buf = io.BytesIO()
            pil_image.save(buf, "JPEG", quality=self.cfg.jpeg_quality)
            buf.seek(0)
            return Image.open(buf).convert("RGB")
        except Exception:
            return pil_image

    # ---- step 9-10: the tensor

    def to_input(self, pil_image):
        """→ float32 CHW, ImageNet-normalised. Identical to production."""
        from PIL import Image
        img = pil_image.convert("RGB").resize(
            (self.cfg.input_size, self.cfg.input_size), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = (arr - np.array(self.cfg.mean, dtype=np.float32)) \
            / np.array(self.cfg.std, dtype=np.float32)
        return arr.transpose(2, 0, 1)

    # ---- the two paths, and there are only two

    def baseline_tensor(self, pil_image, already_cropped=False):
        """**The evaluation transform.** Fully deterministic, no randomness.

        Use this for validation, for benchmark evaluation and for the sealed
        test set. `already_cropped=True` skips detection, for a corpus whose
        crops were extracted ahead of time — the crop is the same either way.
        """
        crop = pil_image if already_cropped else self.detect_face(pil_image).crop
        return self.to_input(self.normalize_compression(crop))

    def training_tensor(self, pil_image, augment, already_cropped=False):
        """The same path with the augmentation layer inserted.

        Augmentation sits between the crop and the q88 tail, so q88 remains
        the last image-domain operation exactly as in production. Passing
        `augment=None` gives `baseline_tensor` back."""
        crop = pil_image if already_cropped else self.detect_face(pil_image).crop
        if augment is not None:
            crop = augment(crop)
        return self.to_input(self.normalize_compression(crop))


# ------------------------------------------------- training-only augmentation

class TrainingAugmentation:
    """Random augmentation, applied **only** during training.

    Operates on the cropped PIL image and returns a PIL image, so the
    production tail (q88 → resize → normalise) runs afterwards untouched.

    The two anti-shortcut transforms are not decoration. V1 reached 96.94%
    having learned the dataset's own resize and JPEG signature rather than
    generator artefacts; randomising both is what stopped that.

    Deliberately absent: MixUp and CutMix. Blending a real face with a fake
    one produces an image whose true label is undefined, and this task's
    decision boundary is exactly what that would blur.
    """

    def __init__(self, config=PRODUCTION, seed=None,
                 rescale_p=0.5, rescale_lo=0.5,
                 jpeg_p=0.9, jpeg_quality=(30, 95),
                 crop_scale=(0.8, 1.0), flip_p=0.5,
                 jitter=(0.15, 0.15, 0.1), grayscale_p=0.05, blur_p=0.2):
        import random
        self.cfg = config
        self.rng = random.Random(seed)
        self.rescale_p, self.rescale_lo = rescale_p, rescale_lo
        self.jpeg_p, self.jpeg_quality = jpeg_p, jpeg_quality
        self.crop_scale, self.flip_p = crop_scale, flip_p
        self.jitter, self.grayscale_p, self.blur_p = jitter, grayscale_p, blur_p

    def __call__(self, img):
        from PIL import Image, ImageEnhance, ImageFilter

        # Resolution fingerprint: down then back up
        if self.rng.random() < self.rescale_p:
            w, h = img.size
            s = self.rng.uniform(self.rescale_lo, 1.0)
            img = img.resize((max(32, int(w * s)), max(32, int(h * s))),
                             Image.BILINEAR).resize((w, h), Image.BILINEAR)

        # Compression fingerprint: a random quality, before the fixed q88
        if self.rng.random() < self.jpeg_p:
            buf = io.BytesIO()
            img.convert("RGB").save(
                buf, "JPEG", quality=self.rng.randint(*self.jpeg_quality))
            buf.seek(0)
            img = Image.open(buf).convert("RGB")

        # Geometry. The crop is already tight around the face, so the scale
        # floor is 0.8 rather than the 0.7 used on uncropped images.
        w, h = img.size
        scale = self.rng.uniform(*self.crop_scale)
        cw, ch = max(16, int(w * scale)), max(16, int(h * scale))
        left = self.rng.randint(0, max(0, w - cw))
        top = self.rng.randint(0, max(0, h - ch))
        img = img.crop((left, top, left + cw, top + ch))

        if self.rng.random() < self.flip_p:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

        b, c, s = self.jitter
        for enhancer, amount in ((ImageEnhance.Brightness, b),
                                 (ImageEnhance.Contrast, c),
                                 (ImageEnhance.Color, s)):
            if amount:
                img = enhancer(img).enhance(1.0 + self.rng.uniform(-amount, amount))

        if self.rng.random() < self.grayscale_p:
            img = img.convert("L").convert("RGB")

        if self.rng.random() < self.blur_p:
            img = img.filter(ImageFilter.GaussianBlur(radius=1))

        return img.convert("RGB")


# ------------------------------------------------------------ torch adapters

def make_transforms(yunet_path, config=PRODUCTION, seed=None):
    """→ (train_fn, eval_fn), both PIL → float32 CHW numpy.

    Wrap in `torch.from_numpy` inside the Dataset. Returning numpy keeps
    this module importable without torch, so the parity test and the
    benchmark can use it on a machine that has no PyTorch — which is the
    normal state of a DeepShield deployment."""
    pre = Preprocessor(yunet_path, config)
    aug = TrainingAugmentation(config, seed=seed)

    def train_fn(pil_image, already_cropped=True):
        return pre.training_tensor(pil_image, aug, already_cropped=already_cropped)

    def eval_fn(pil_image, already_cropped=True):
        return pre.baseline_tensor(pil_image, already_cropped=already_cropped)

    return train_fn, eval_fn
