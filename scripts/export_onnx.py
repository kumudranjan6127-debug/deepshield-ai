"""Export a trained checkpoint to a single-file ONNX model.

    venv\\Scripts\\python scripts/export_onnx.py

Reads models/deepshield_mobilenetv3.pth and writes deepshield.onnx beside
it, plus deepshield.onnx.json with the metadata the runtime needs (classes,
input size, normalisation, accuracies). With those two files the backend no
longer needs PyTorch — OpenCV's DNN module runs the network.

Requires torch, so run it once after training; deployment does not.
"""
import json
import os
import sys

import torch
from torchvision import models

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = os.path.join(ROOT, "models", "deepshield_mobilenetv3.pth")
ONNX = os.path.join(ROOT, "models", "deepshield.onnx")
META = ONNX + ".json"

BUILDERS = {
    "mobilenet_v3_small": models.mobilenet_v3_small,
    "mobilenet_v3_large": models.mobilenet_v3_large,
}


def main():
    if not os.path.exists(CKPT):
        sys.exit(f"checkpoint not found: {CKPT}")

    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    arch = ck.get("arch", "mobilenet_v3_small")
    if arch not in BUILDERS:
        sys.exit(f"unsupported arch: {arch}")

    classes = ck["classes"]
    net = BUILDERS[arch](weights=None)
    net.classifier[3] = torch.nn.Linear(net.classifier[3].in_features, len(classes))
    net.load_state_dict(ck["state_dict"])
    net.eval()

    size = ck.get("input_size", 224)
    # dynamo=False keeps the weights inside the file instead of a side-car
    torch.onnx.export(
        net, torch.randn(1, 3, size, size), ONNX,
        input_names=["input"], output_names=["logits"],
        opset_version=17, dynamo=False,
    )

    meta = {
        "arch": arch,
        "classes": classes,
        "input_size": size,
        "normalize": ck.get("normalize", {"mean": [0.485, 0.456, 0.406],
                                          "std": [0.229, 0.224, 0.225]}),
        "val_accuracy": ck.get("val_accuracy"),
        "robust_val_accuracy": ck.get("robust_val_accuracy"),
        "tpdn_accuracy": ck.get("tpdn_accuracy"),
        "dfdc_accuracy": ck.get("dfdc_accuracy"),
        "test_accuracy": ck.get("test_accuracy"),
        "trained_on": ck.get("trained_on"),
        "source_checkpoint": os.path.basename(CKPT),
    }
    with open(META, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"wrote {os.path.basename(ONNX)}  {os.path.getsize(ONNX)/1e6:.1f} MB")
    print(f"wrote {os.path.basename(META)}  ({arch}, classes={classes})")

    # Prove the exported graph reproduces the checkpoint before anyone ships it
    import cv2
    import numpy as np
    dnn = cv2.dnn.readNetFromONNX(ONNX)
    x = torch.randn(2, 3, size, size)
    with torch.no_grad():
        a = torch.softmax(net(x), dim=1).numpy()
    dnn.setInput(x.numpy())
    logits = dnn.forward()
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    b = e / e.sum(axis=1, keepdims=True)
    diff = float(np.abs(a - b).max())
    print(f"parity vs PyTorch: max difference {diff:.2e}")
    if diff > 1e-4:
        sys.exit("ONNX output differs from PyTorch — not safe to ship")
    print("OK — the ONNX model matches the checkpoint")


if __name__ == "__main__":
    main()
