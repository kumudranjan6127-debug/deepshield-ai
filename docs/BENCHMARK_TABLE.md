| Metric | Value | Basis |
|---|---|---|
| Images scored | **592** | 524 real, 68 fake |
| Independent groups | **512** | the honest sample size — images inside a group are correlated |
| Accuracy | **100.00%** | at threshold 0.50 |
| Precision | **100.00%** | of everything called fake |
| Recall | **100.00%** | of the fakes present |
| F1 | **100.00%** | |
| Specificity | **100.00%** | of the real images |
| ROC-AUC | **1.0000** | ranking quality |
| PR-AUC | **1.0000** | |
| **False-positive rate** | **0.00%** | **a real photograph called fake** |
| False-negative rate | **0.00%** | a deepfake called real |
| Brier score | **0.0006** | calibration, 0 is perfect |
| ECE | **0.0242** | calibration error |
| Cross-dataset accuracy | *not measured* | needs `--seen` and a generator outside it |

Per source:

| Class | Source | n | Mean P(fake) | Outcome |
|---|---|---|---|---|
| fake | `orig` | 5 | 0.975 | 100.00% detected |
| fake | `phone` | 5 | 0.974 | 100.00% detected |
| fake | `reencode` | 5 | 0.975 | 100.00% detected |
| fake | `screenshot` | 5 | 0.974 | 100.00% detected |
| fake | `social` | 5 | 0.975 | 100.00% detected |
| fake | `synthetic_clip` | 38 | 0.976 | 100.00% detected |
| fake | `tpdn` | 5 | 0.974 | 100.00% detected |
| real | `lfw` | 500 | 0.024 | 0.00% called fake |
| real | `portrait` | 24 | 0.022 | 0.00% called fake |
