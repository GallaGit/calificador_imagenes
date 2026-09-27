"""Build and verify the EfficientNet-B0 cats-vs-dogs classifier."""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D


IMAGE_SIZE = (224, 224)
NUM_CLASSES = 2


def build_model() -> tf.keras.Model:
    """Create and compile the binary image classifier."""

    # EfficientNet extracts visual features from each image. We remove its
    # original ImageNet classifier because this project has only two classes.
    backbone = EfficientNetB0(
        include_top=False,
        weights=None,
        input_shape=(*IMAGE_SIZE, 3),
    )

    model = Sequential(
        [
            backbone,
            # Summarize the feature maps into one vector per image.
            GlobalAveragePooling2D(),
            # Learn combinations of the visual features.
            Dense(128, activation="relu"),
            # Softmax returns one probability for each class: cat and dog.
            Dense(NUM_CLASSES, activation="softmax"),
        ],
        name="cats_vs_dogs_efficientnet",
    )

    # Adam updates the model weights. Categorical crossentropy matches the
    # one-hot labels produced by the categorical image generators.
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main() -> None:
    model = build_model()
    model.summary()

    # Verify that the model accepts the same shape as the prepared dataset.
    test_batch = tf.zeros((2, *IMAGE_SIZE, 3), dtype=tf.float32)
    predictions = model(test_batch)

    print(f"Entrada de prueba: {test_batch.shape}")
    print(f"Salida del modelo: {predictions.shape}")
    print(f"Dispositivos GPU detectados: {tf.config.list_physical_devices('GPU')}")

    if tuple(predictions.shape) != (2, NUM_CLASSES):
        raise RuntimeError(
            f"Salida inesperada: se esperaba (2, {NUM_CLASSES}), "
            f"pero se obtuvo {predictions.shape}"
        )

    print("Verificacion del modelo completada correctamente.")


if __name__ == "__main__":
    main()
