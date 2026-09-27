"""Explore and validate the Dogs vs Cats image dataset."""

from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image, UnidentifiedImageError


IMAGE_PATTERN = re.compile(r"^(cat|dog)\.(\d+)\.jpg$", re.IGNORECASE)
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("data/raw/train"),
        help="Directory containing the original labelled images.",
    )
    parser.add_argument(
        "--figure-path",
        type=Path,
        default=Path("docs/figures/dataset-samples.png"),
        help="Path where the sample figure will be saved.",
    )
    return parser.parse_args()


def collect_images(dataset_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in dataset_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def validate_images(image_paths: list[Path]) -> tuple[Counter[str], Counter[tuple[int, int]], list[str]]:
    labels: Counter[str] = Counter()
    dimensions: Counter[tuple[int, int]] = Counter()
    invalid_files: list[str] = []

    for image_path in image_paths:
        match = IMAGE_PATTERN.match(image_path.name)
        if not match:
            invalid_files.append(f"{image_path.name}: nombre no reconocido")
            continue

        label = match.group(1).lower()
        try:
            with Image.open(image_path) as image:
                image.verify()
            with Image.open(image_path) as image:
                labels[label] += 1
                dimensions[image.size] += 1
        except (OSError, UnidentifiedImageError) as error:
            invalid_files.append(f"{image_path.name}: {error}")

    return labels, dimensions, invalid_files


def create_sample_figure(image_paths: list[Path], figure_path: Path) -> None:
    figure, axes = plt.subplots(2, 9, figsize=(18, 5))
    figure.suptitle("Muestras del dataset Dogs vs Cats")

    for row, label in enumerate(("cat", "dog")):
        samples = [
            path for path in image_paths if path.name.lower().startswith(f"{label}.")
        ][:9]
        for column, axis in enumerate(axes[row]):
            axis.axis("off")
            if column >= len(samples):
                continue
            with Image.open(samples[column]) as image:
                axis.imshow(image.convert("RGB"))
            axis.set_title(label)

    figure.tight_layout()
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(figure_path, dpi=150)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if not args.dataset_dir.is_dir():
        raise FileNotFoundError(f"No existe el directorio: {args.dataset_dir}")

    image_paths = collect_images(args.dataset_dir)
    labels, dimensions, invalid_files = validate_images(image_paths)
    create_sample_figure(image_paths, args.figure_path)

    print(f"Directorio: {args.dataset_dir}")
    print(f"Imagenes encontradas: {len(image_paths)}")
    print(f"Etiquetas: {dict(labels)}")
    print(f"Tamanos distintos: {len(dimensions)}")
    print(f"Tamano mas frecuente: {dimensions.most_common(1)}")
    print(f"Archivos no validos: {len(invalid_files)}")
    print(f"Figura guardada en: {args.figure_path}")

    if invalid_files:
        print("Detalles de archivos no validos:")
        for invalid_file in invalid_files:
            print(f"- {invalid_file}")
        raise SystemExit(1)

    if labels != Counter({"cat": 12500, "dog": 12500}):
        raise SystemExit("Las cantidades esperadas son 12.500 gatos y 12.500 perros.")


if __name__ == "__main__":
    main()
