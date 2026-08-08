# models/archive/ — model version history

Every trained model, kept so any version can be restored instantly.
The **live** model is always `models/deepshield_mobilenetv3.pth`.

| File | Arch | Test acc | Robust val | Trained on |
|---|---|---|---|---|
| `v1_baseline.pth` | MobileNetV3-Small | 96.94% | — | 25k/class, 3 epochs, light aug |
| `v2_heavy.pth` | MobileNetV3-Small | **99.39%** | **98.54%** | 50k/class, 10 epochs, anti-shortcut aug, robust-selected |
| *(v3_max.pth)* | MobileNetV3-Large | — | — | multi-generator: SG1 + TPDN/SG2 + diffusion *(not trained yet)* |

## Notes

- **V1 → V2 lesson:** V1 scored well in-dataset but learned the dataset's
  JPEG/resize *pipeline fingerprint* rather than real GAN artifacts, so it
  called off-pipeline fakes "real". V2 added randomized JPEG/rescale/blur
  augmentation and selects its checkpoint by **robust** accuracy.
- **Wild test (V2):** 2 of 3 thispersondoesnotexist (StyleGAN2) faces were
  caught. V3 mixes StyleGAN2 + diffusion fakes into training to close that
  cross-generator gap.

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
