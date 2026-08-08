# models/archive/ — model version history

Every trained model, kept so any version can be restored instantly.
The **live** model is always `models/deepshield_mobilenetv3.pth`.

| File | Arch | Val | Robust | TPDN holdout | Trained on |
|---|---|---|---|---|---|
| `v1_baseline.pth` | MobileNetV3-Small | — | — | — | 25k/class, 3 epochs, light aug (96.94% test) |
| `v2_heavy.pth` | MobileNetV3-Small | 99.40% | 98.54% | — | 50k/class, 10 epochs, anti-shortcut aug (99.39% test) |
| `v3_max.pth` | **MobileNetV3-Large** | **99.90%** | **99.18%** | **100.00%** | multi-generator: StyleGAN1 + StyleGAN2/TPDN + diffusion, 10 epochs |

## What each version taught us

- **V1 → V2:** V1 scored well in-dataset but had learned the dataset's
  JPEG/resize *pipeline fingerprint*, not real GAN artefacts, so it called
  off-pipeline fakes "real". V2 added randomized JPEG/rescale/blur
  augmentation and picks its checkpoint by **robust** accuracy.
- **V2 → V3:** V2 still only knew StyleGAN1. Real thispersondoesnotexist
  (StyleGAN2) faces scored 0.02–0.49. V3 trains on three generator
  families at once and scores those same faces 0.97–0.98.
- **V3 exposed two preprocessing bugs** (fixed in `backend/inference.py`):
  inputs are capped at 1024px, and our model's input goes through one
  JPEG q88 round-trip, because a 2687px pristine original scored 0.94
  fake while the same photo re-saved scored 0.02.

Current measured score on the held images: **9/9** (5 StyleGAN2 fakes +
an authentic press portrait at four resolutions).

## Switching versions

```bash
# roll back to any archived version
cp models/archive/v2_heavy.pth models/deepshield_mobilenetv3.pth
```

The backend hot-reloads on file change — no restart needed. The checkpoint
carries its own `arch`, so Small and Large models are both loadable.

## Promoting a new model

1. Save the new checkpoint here as `vN_<name>.pth` **first**.
2. Copy it over `models/deepshield_mobilenetv3.pth` to go live.
3. Compare against the previous version before keeping it — never train
   on top of a live model.
