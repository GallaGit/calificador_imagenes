"""Split the raw dataset and create Keras image generators."""

from __future__ import annotations

import argparse
import random
import shutil
from collections import Counter
from pathlib import Path


LABELS = ("cat", "dog")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=Path("data/raw/train"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--skip-generators",
        action="store_true",
        help="Only create the train/test split without importing TensorFlow.",
    )
    return parser.parse_args()


def images_for_label(source_dir: Path, label: str) -> list[Path]:
    return sorted(
        path
        for path in source_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
        and path.name.lower().startswith(f"{label}.")
    )


def split_images(
    source_dir: Path, output_dir: Path, test_size: float, seed: int
) -> Counter[tuple[str, str]]:
    if not 0 < test_size < 1:
        raise ValueError("test_size debe estar entre 0 y 1")
    if not source_dir.is_dir():
        raise FileNotFoundError(f"No existe el directorio: {source_dir}")

    random_generator = random.Random(seed)
    counts: Counter[tuple[str, str]] = Counter()

    for label in LABELS:
        files = images_for_label(source_dir, label)
        random_generator.shuffle(files)
        test_count = round(len(files) * test_size)
        split_files = {
            "test": files[:test_count],
            "train": files[test_count:],
        }

        for split, selected_files in split_files.items():
            destination = output_dir / split / label
            destination.mkdir(parents=True, exist_ok=True)
            for image_path in selected_files:
                shutil.copy2(image_path, destination / image_path.name)
            counts[(split, label)] = len(selected_files)

    return counts


def create_generators(output_dir: Path, batch_size: int):
    try:
        from tensorflow.keras.preprocessing.image import ImageDataGenerator
    except ImportError as error:
        raise RuntimeError(
            "TensorFlow no esta instalado. Ejecuta: pip install -r requirements.txt"
        ) from error

    train_datagen = ImageDataGenerator(rescale=1.0 / 255)
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)
    common_options = {
        "target_size": (224, 224),
        "batch_size": batch_size,
        "class_mode": "categorical",
        "classes": list(LABELS),
    }

    train_generator = train_datagen.flow_from_directory(
        output_dir / "train", shuffle=True, **common_options
    )
    test_generator = test_datagen.flow_from_directory(
        output_dir / "test", shuffle=False, **common_options
    )
    return train_generator, test_generator


def main() -> None:
    args = parse_args()
    counts = split_images(args.source_dir, args.output_dir, args.test_size, args.seed)

    print(f"Semilla: {args.seed}")
    print(f"Tamano de batch: {args.batch_size}")
    for split in ("train", "test"):
        for label in LABELS:
            print(f"{split}/{label}: {counts[(split, label)]}")

    if args.skip_generators:
        print("Generadores omitidos porque se solicito --skip-generators")
        return

    train_generator, test_generator = create_generators(args.output_dir, args.batch_size)
    train_images, train_labels = next(train_generator)
    test_images, test_labels = next(test_generator)
    print(f"Clases: {train_generator.class_indices}")
    print(f"Batch de entrenamiento: {train_images.shape}, {train_labels.shape}")
    print(f"Batch de prueba: {test_images.shape}, {test_labels.shape}")


if __name__ == "__main__":
    main()
