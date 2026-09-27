"""Train, optimize, evaluate, and export the cats-vs-dogs classifier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from build_model import IMAGE_SIZE, build_model
from prepare_dataset import LABELS


ROOT_DIR = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT_DIR / "data/processed")
    parser.add_argument("--models-dir", type=Path, default=ROOT_DIR / "models")
    parser.add_argument("--artifacts-dir", type=Path, default=ROOT_DIR / "artifacts")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--baseline-epochs", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def create_generators(data_dir: Path, batch_size: int, seed: int):
    try:
        from tensorflow.keras.preprocessing.image import ImageDataGenerator
    except ImportError as error:
        raise RuntimeError(
            "TensorFlow no esta instalado. Ejecuta: pip install -r requirements.txt"
        ) from error

    # Keep 10% of the training directory for validation. Test remains untouched
    # until the final evaluation, so it is not used to choose checkpoints.
    train_datagen = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.1)
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)
    options = {
        "target_size": IMAGE_SIZE,
        "batch_size": batch_size,
        "class_mode": "categorical",
        "classes": list(LABELS),
    }
    train = train_datagen.flow_from_directory(
        data_dir / "train", subset="training", shuffle=True, seed=seed, **options
    )
    validation = train_datagen.flow_from_directory(
        data_dir / "train", subset="validation", shuffle=False, seed=seed, **options
    )
    test = test_datagen.flow_from_directory(
        data_dir / "test", shuffle=False, **options
    )
    return train, validation, test


def history_dict(history) -> dict[str, list[float]]:
    return {key: [float(value) for value in values] for key, values in history.history.items()}


def merge_histories(*histories: dict[str, list[float]]) -> dict[str, list[float]]:
    merged: dict[str, list[float]] = {}
    for history in histories:
        for key, values in history.items():
            merged.setdefault(key, []).extend(values)
    return merged


def save_training_plot(history: dict[str, list[float]], output_path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    for axis, metric, title in zip(
        axes, ("loss", "accuracy"), ("Perdida", "Accuracy")
    ):
        axis.plot(history.get(metric, []), label="entrenamiento")
        axis.plot(history.get(f"val_{metric}", []), label="validacion")
        axis.set_title(title)
        axis.set_xlabel("Epoca")
        axis.legend()
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def evaluate_and_save(model, test_generator, artifacts_dir: Path) -> dict[str, object]:
    from sklearn.metrics import classification_report, confusion_matrix

    test_generator.reset()
    probabilities = model.predict(test_generator, verbose=0)
    predictions = np.argmax(probabilities, axis=1)
    labels = test_generator.classes
    matrix = confusion_matrix(labels, predictions, labels=range(len(LABELS)))
    report = classification_report(
        labels, predictions, labels=range(len(LABELS)), target_names=LABELS,
        output_dict=True, zero_division=0,
    )

    figure, axis = plt.subplots(figsize=(5, 4))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set(xticks=range(len(LABELS)), yticks=range(len(LABELS)),
             xticklabels=LABELS, yticklabels=LABELS, xlabel="Prediccion", ylabel="Real")
    for row in range(len(LABELS)):
        for column in range(len(LABELS)):
            axis.text(column, row, matrix[row, column], ha="center", va="center")
    figure.tight_layout()
    figure.savefig(artifacts_dir / "confusion_matrix.png", dpi=150)
    plt.close(figure)

    evaluation = model.evaluate(test_generator, verbose=0)
    metrics = {"loss": float(evaluation[0]),
               "accuracy": float(evaluation[1]),
               "classification_report": report, "confusion_matrix": matrix.tolist()}
    return metrics


def train(args: argparse.Namespace) -> dict[str, object]:
    import tensorflow as tf

    if args.baseline_epochs < 0 or args.epochs < 1 or args.patience < 0:
        raise ValueError("Las epocas deben ser validas y patience no puede ser negativo")
    tf.keras.utils.set_random_seed(args.seed)
    args.models_dir.mkdir(parents=True, exist_ok=True)
    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    train_generator, validation_generator, test_generator = create_generators(
        args.data_dir, args.batch_size, args.seed
    )
    model = build_model()
    baseline = model.fit(
        train_generator, validation_data=validation_generator,
        epochs=args.baseline_epochs, verbose=1,
    ) if args.baseline_epochs else None
    best_path = args.models_dir / "best_model.keras"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            best_path, monitor="val_accuracy", mode="max", save_best_only=True
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=args.patience, restore_best_weights=True
        ),
    ]
    optimized = model.fit(
        train_generator, validation_data=validation_generator,
        epochs=args.epochs, callbacks=callbacks, verbose=1,
    )
    model = tf.keras.models.load_model(best_path)
    model.save(args.models_dir / "cats_vs_dogs.keras")
    history = merge_histories(
        history_dict(baseline) if baseline else {}, history_dict(optimized)
    )
    (args.artifacts_dir / "history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    save_training_plot(history, args.artifacts_dir / "training_history.png")
    metrics = evaluate_and_save(model, test_generator, args.artifacts_dir)
    metadata = {
        "image_size": list(IMAGE_SIZE), "classes": list(LABELS),
        "class_indices": test_generator.class_indices, "seed": args.seed,
        "batch_size": args.batch_size, "metrics": metrics,
    }
    (args.models_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return metrics


def main() -> None:
    metrics = train(parse_args())
    print(f"Accuracy en prueba: {metrics['accuracy']:.4f}")
    print("Modelo y artefactos guardados correctamente.")


if __name__ == "__main__":
    main()
