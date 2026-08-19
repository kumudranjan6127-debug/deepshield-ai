# Third-party model provenance

DeepShield keeps externally trained models separate from models trained by this project. A third-party score is supporting evidence, not a DeepShield accuracy claim.

## AI Image Detect Distilled — INT8 ONNX

| Field | Value |
|---|---|
| Purpose | Full-frame AI-generated image/frame evidence |
| Source repository | `onnx-community/ai-image-detect-distilled-ONNX` |
| Upstream license | MIT (as declared by the upstream model repository) |
| Pinned revision | `7f067e23521eeb6d6525221af82c613fb746aaff` |
| File | `onnx/model_int8.onnx` |
| Expected SHA-256 | `7273cb9cd81e17eae04771010d2199ba6ae34ea2a75a275518c0bc4a2c26ffd2` |
| Input | RGB 224×224; rescale 1/255; normalize mean/std 0.5 |
| Output convention | class 0 = `fake`, class 1 = `real` |
| DeepShield calibration | **Not calibrated** |

The runtime never downloads this model during a user request. `scripts/fetch_origin_model.py` downloads the pinned artifact during setup/deploy, verifies the SHA-256 digest, and only then installs it as `models/ai_origin_int8.onnx`.

### Product policy

- The V3 DeepShield face detector remains an independent signal.
- The full-frame detector is intended to catch fully synthetic media that a face-only crop can miss.
- Its score is not presented as a measured probability of AI generation.
- The default decision threshold is intentionally conservative until DeepShield has a sufficiently large sealed real/fake benchmark for calibration.
- Upstream benchmark claims are not copied into DeepShield's own model card as DeepShield performance.

Before changing the pinned revision, file, preprocessing, class mapping, or threshold, update the provenance entry and re-run the held-out evaluation protocol.
