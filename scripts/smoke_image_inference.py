"""CPU smoke test: load image.h5 and run one real prediction."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.detector import DetectorError, load_image_model, predict_image_file  # noqa: E402


def main() -> int:
    assets = ROOT / "assets"
    assets.mkdir(exist_ok=True)
    image_path = assets / "smoke_test.png"

    # Synthetic RGB image is enough to prove the model actually runs.
    rng = np.random.default_rng(0)
    array = rng.integers(0, 256, size=(224, 224, 3), dtype=np.uint8)
    Image.fromarray(array, mode="RGB").save(image_path)

    print("Python:", sys.version)
    print("Image:", image_path)

    try:
        model = load_image_model()
        print("Model loaded:", type(model).__name__)
        print("Input shape:", getattr(model, "input_shape", None))
        print("Output shape:", getattr(model, "output_shape", None))
        result = predict_image_file(image_path, model)
    except DetectorError as exc:
        print("DETECTOR ERROR:", exc)
        return 1

    print("Label:", result["label"])
    print("Deepfake score:", f"{result['deepfake_score']:.6f}")
    print("Architecture:", result["architecture"])
    print("Color space:", result["color_space"])
    print("Input size:", result["input_size"])
    print("SMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
