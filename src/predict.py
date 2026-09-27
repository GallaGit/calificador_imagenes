"""Classify one image using the exported model."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from build_model import IMAGE_SIZE


ROOT_DIR = Path(__file__).resolve().parent.parent


def predict_image(image_path: Path, model_path: Path) -> tuple[str, float]:
    import tensorflow as tf

    if not image_path.is_file():
        raise FileNotFoundError(f"No existe la imagen: {image_path}")
    model = tf.keras.models.load_model(model_path)
    with Image.open(image_path) as image:
        array = np.asarray(image.convert("RGB").resize(IMAGE_SIZE), dtype=np.float32) / 255
    probabilities = model.predict(array[None, ...], verbose=0)[0]
    index = int(np.argmax(probabilities))
    labels = ("cat", "dog")
    return labels[index], float(probabilities[index])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--model", type=Path, default=ROOT_DIR / "models/cats_vs_dogs.keras")
    args = parser.parse_args()
    label, confidence = predict_image(args.image, args.model)
    print(f"Prediccion: {label}")
    print(f"Confianza: {confidence:.2%}")


if __name__ == "__main__":
    main()
