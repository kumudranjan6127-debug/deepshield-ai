"""Run one image through the real pipeline and show every face separately.

    venv/Scripts/python scripts/check_case.py testcases/whatever.jpg

The point is the per-face table. A verdict of "real" on a picture with a
swapped face in it has two very different causes - the swapped face was
never scored, or it was scored and read as real - and only the per-face
numbers tell them apart.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "backend"))


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    path = sys.argv[1]
    if not os.path.exists(path):
        print(f"not found: {path}")
        return 2

    from PIL import Image
    import inference

    engine = inference._get_engine()
    fake_i = engine.classes.index("fake")

    with Image.open(path) as im:
        print(f"\n{os.path.basename(path)}   {im.size[0]}x{im.size[1]}")
        detections = engine._detect_faces(im)

    print("=" * 58)
    if not detections[0]["found"]:
        print("  no face detected - the whole frame was scored")
    else:
        print(f"  {'face':<6}{'crop':<12}{'area':>8}   P(fake)")
        print("  " + "-" * 40)
        for i, d in enumerate(detections):
            p = float(engine._probs_raw(d["crop"])[fake_i])
            w, h = d["crop"].size
            flag = "  <-- most suspicious" if p == max(
                float(engine._probs_raw(x["crop"])[fake_i]) for x in detections) else ""
            print(f"  {i:<6}{f'{w}x{h}':<12}"
                  f"{int(d['box'][2] * d['box'][3]):>8}   {p:.3f}{flag}")

    result = inference.analyze_file(path, "image")
    print("\n  verdict:  "
          f"{result['prediction'].upper()} {result['confidence']}%   "
          f"faces={result.get('facesFound')}  faceFound={result.get('faceFound')}")

    if result["prediction"] == "real":
        print("\n  If every face above scored low, this is not the multi-face")
        print("  bug - it is KNOWN_ISSUES #3, the face-swap gap. The model was")
        print("  never trained on swaps of this kind and does not see them.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
