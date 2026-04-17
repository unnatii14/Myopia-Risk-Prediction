import os
import traceback
import numpy as np

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
IMAGE_MODEL_PATH = os.path.join(MODEL_DIR, "myopia_image_classifier.keras")


def main() -> int:
    print(f"Checking image model path: {IMAGE_MODEL_PATH}")
    if not os.path.isdir(IMAGE_MODEL_PATH):
        print("[FAIL] Model directory not found.")
        return 1

    try:
        import tensorflow as tf
    except Exception as exc:
        print(f"[FAIL] TensorFlow import failed: {exc}")
        return 1

    try:
        model = tf.keras.models.load_model(IMAGE_MODEL_PATH)
        print("[OK] Model loaded successfully")
        print(f"[INFO] Input shape: {model.input_shape}")
        print(f"[INFO] Output shape: {model.output_shape}")

        # Dry-run inference with random image tensor matching 224x224x3.
        x = np.random.randint(0, 256, size=(1, 224, 224, 3), dtype=np.uint8).astype(np.float32)
        y = model.predict(x, verbose=0)
        print(f"[OK] Inference works. Sample output: {np.squeeze(y)}")
        return 0
    except Exception as exc:
        print(f"[FAIL] Model load/inference failed: {exc}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
