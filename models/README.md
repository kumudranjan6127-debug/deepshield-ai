# models/

Drop the trained checkpoint here:

```
models/deepshield_mobilenetv3.pth
```

It is produced by `training/DeepShield_Training_Colab.ipynb` (run on Google Colab, free T4 GPU).
The Flask backend (`app.py`) loads it in Phase 4; until then the API returns simulated verdicts.

This folder is blocked from being served over HTTP (see `BLOCKED_DIRS` in app.py).
