"""
Diagnose whether the image classifier learned real retinal signs or a
dataset shortcut (source leakage).

Run from the backend folder with your venv active:

    python diagnose_image_model.py --data "C:/Users/admin/Desktop/6th sem/Project-III/IMAGES"

It runs four checks and prints a verdict:

  1. FILE PROPERTIES  - do the two class folders differ in format / size /
                        dimensions? (different sources = leakage risk)
  2. CHANNEL STATS    - systematic colour/brightness gap between classes?
  3. CENTRE-MASK TEST - black out the retina, keep only the border/background.
                        If the model still classifies well, it is reading
                        non-retinal artifacts, NOT the eye.
  4. BORDER-MASK TEST - keep only the retina, black out everything else.
                        Accuracy should HOLD here if the model is legit.

A trustworthy model: fails check 3 (can't tell without the retina) and
passes check 4. A leaking model: still "works" in check 3.
"""
import argparse
import os
import glob
import collections
import numpy as np
from PIL import Image

IMG = 224


def load_model(models_dir):
    """Prefer ONNX (matches the API); fall back to Keras."""
    onnx_path = os.path.join(models_dir, "myopia_classifier.onnx")
    if os.path.isfile(onnx_path):
        import onnxruntime as ort
        sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        name = sess.get_inputs()[0].name

        def predict(batch):  # batch: (N,224,224,3) float32
            return np.squeeze(sess.run(None, {name: batch})[0])
        return predict, "onnx"

    keras_path = os.path.join(models_dir, "myopia_image_classifier.keras")
    import tensorflow as tf
    model = tf.keras.models.load_model(keras_path)

    def predict(batch):
        return np.squeeze(model.predict(batch, verbose=0))
    return predict, "keras"


def myopia_prob(raw):
    """API convention: sigmoid output = P(NORMAL), so myopia = 1 - raw."""
    raw = np.clip(raw, 0.0, 1.0)
    return 1.0 - raw


def find_class_dirs(data_dir):
    subs = [d for d in sorted(os.listdir(data_dir))
            if os.path.isdir(os.path.join(data_dir, d))]
    myo = next((d for d in subs if "myop" in d.lower()), None)
    nor = next((d for d in subs if "norm" in d.lower()), None)
    if not myo or not nor:
        raise SystemExit(f"Could not find Myopia/Normal folders in {data_dir}. Found: {subs}")
    return os.path.join(data_dir, myo), os.path.join(data_dir, nor)


def sample_files(folder, n):
    files = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG"):
        files += glob.glob(os.path.join(folder, ext))
    files.sort()
    if len(files) > n:
        idx = np.linspace(0, len(files) - 1, n).astype(int)
        files = [files[i] for i in idx]
    return files


def file_properties(files):
    fmt = collections.Counter()
    sizes, dims = [], collections.Counter()
    for f in files:
        try:
            im = Image.open(f)
            fmt[im.format] += 1
            dims[im.size] += 1
            sizes.append(os.path.getsize(f) / 1024)
        except Exception:
            pass
    return fmt, dims, np.array(sizes)


def to_batch(files, mask=None):
    out = []
    for f in files:
        im = Image.open(f).convert("RGB").resize((IMG, IMG))
        a = np.asarray(im, np.float32)
        if mask == "centre":      # black out the retina (centre disc)
            yy, xx = np.ogrid[:IMG, :IMG]
            disc = (xx - IMG / 2) ** 2 + (yy - IMG / 2) ** 2 <= (IMG * 0.42) ** 2
            a[disc] = 0
        elif mask == "border":    # keep only the retina, black out the rest
            yy, xx = np.ogrid[:IMG, :IMG]
            disc = (xx - IMG / 2) ** 2 + (yy - IMG / 2) ** 2 <= (IMG * 0.42) ** 2
            a[~disc] = 0
        out.append(a)
    return np.stack(out).astype(np.float32)


def accuracy(predict, files_myo, files_nor, mask=None):
    pm = myopia_prob(predict(to_batch(files_myo, mask)))
    pn = myopia_prob(predict(to_batch(files_nor, mask)))
    correct = int((pm >= 0.5).sum() + (pn < 0.5).sum())
    total = len(pm) + len(pn)
    return correct / total, float(pm.mean()), float(pn.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to IMAGES folder")
    ap.add_argument("--models", default="../models", help="Path to models folder")
    ap.add_argument("--n", type=int, default=100, help="Images sampled per class")
    args = ap.parse_args()

    myo_dir, nor_dir = find_class_dirs(args.data)
    fm = sample_files(myo_dir, args.n)
    fn = sample_files(nor_dir, args.n)
    print(f"Sampled {len(fm)} myopia + {len(fn)} normal images\n")

    # 1. FILE PROPERTIES
    print("=" * 60)
    print("1. FILE PROPERTIES  (differences here = possible source leakage)")
    for label, files in [("MYOPIA", fm), ("NORMAL", fn)]:
        fmt, dims, sizes = file_properties(files)
        top_dim = dims.most_common(1)[0] if dims else ("-", 0)
        print(f"  {label}: formats={dict(fmt)}  "
              f"most-common-size={top_dim[0]} ({top_dim[1]}/{len(files)})  "
              f"KB mean={sizes.mean():.0f} std={sizes.std():.0f}")

    # 2. CHANNEL STATS
    print("=" * 60)
    print("2. CHANNEL / BRIGHTNESS STATS")
    for label, files in [("MYOPIA", fm), ("NORMAL", fn)]:
        vals = []
        for f in files:
            a = np.asarray(Image.open(f).convert("RGB").resize((IMG, IMG)), np.float32)
            m = a.mean(2) > 20
            if m.any():
                vals.append([a[..., i][m].mean() for i in range(3)])
        v = np.array(vals).mean(0)
        print(f"  {label}: R={v[0]:6.1f}  G={v[1]:6.1f}  B={v[2]:6.1f}")

    # 3 & 4. MASK TESTS
    predict, kind = load_model(args.models)
    print("=" * 60)
    print(f"3 & 4. MASK TESTS  (model backend: {kind})")
    acc_full, _, _ = accuracy(predict, fm, fn)
    acc_ctr, _, _ = accuracy(predict, fm, fn, mask="centre")
    acc_brd, _, _ = accuracy(predict, fm, fn, mask="border")
    print(f"  full image accuracy        : {acc_full*100:5.1f}%")
    print(f"  retina BLACKED OUT accuracy : {acc_ctr*100:5.1f}%   (should drop to ~50%)")
    print(f"  retina ONLY accuracy        : {acc_brd*100:5.1f}%   (should stay high)")

    # VERDICT
    print("=" * 60)
    print("VERDICT")
    if acc_ctr > 0.70:
        print("  ⚠ LEAKAGE LIKELY: the model classifies well even with the retina")
        print("    blacked out — it is reading background/border artifacts, not the eye.")
    elif acc_brd < 0.70 <= acc_full:
        print("  ⚠ ODD: accuracy collapses when only the retina is shown. The model")
        print("    may rely on the border region rather than retinal detail.")
    else:
        print("  ✓ Looks reasonable: needs the retina to classify, and the border")
        print("    alone isn't enough. Still confirm both folders share one source.")


if __name__ == "__main__":
    main()
