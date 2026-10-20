
from pathlib import Path
import json

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)

from training.data_loader import (
    prepare_datasets,
    CLASS_NAMES,
    IMAGE_SIZE,
)

SEED = 42
EPOCHS = 15
LEARNING_RATE = 1e-4
DROPOUT_RATE = 0.4

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL_PATH = MODEL_DIR / "vgg16_brisc_best.keras"
HISTORY_PATH = OUTPUT_DIR / "vgg16_training_history.json"

tf.keras.utils.set_random_seed(SEED)


def build_model():
    """Build a four-class VGG16 transfer-learning classifier."""

    base_model = VGG16(
        weights="imagenet",
        include_top=False,
        input_shape=(*IMAGE_SIZE, 3),
    )
    base_model.trainable = False

    inputs = layers.Input(shape=(*IMAGE_SIZE, 3), name="mri_image")
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
    x = layers.Dropout(DROPOUT_RATE, name="dropout")(x)
    outputs = layers.Dense(
        len(CLASS_NAMES),
        activation="softmax",
        name="class_probabilities",
    )(x)

    model = models.Model(inputs, outputs, name="vgg16_brisc_classifier")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=LEARNING_RATE
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def main():
    print("Loading datasets...")
    train_ds, val_ds, _ = prepare_datasets()

    print("\nBuilding VGG16 model...")
    model = build_model()
    model.summary()

    callbacks = [
        ModelCheckpoint(
            filepath=str(BEST_MODEL_PATH),
            monitor="val_loss",
            save_best_only=True,
            mode="min",
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=4,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=2,
            min_lr=1e-7,
            verbose=1,
        ),
    ]

    print("\nStarting training...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1,
    )

    # Save the recorded history for plots and analysis.
    history_data = {
        key: [float(value) for value in values]
        for key, values in history.history.items()
    }

    with HISTORY_PATH.open("w", encoding="utf-8") as file:
        json.dump(history_data, file, indent=2)

    print("\nTraining finished.")
    print(f"Best model checkpoint: {BEST_MODEL_PATH}")
    print(f"Training history: {HISTORY_PATH}")
    print("Recorded metrics:", list(history_data.keys()))


if __name__ == "__main__":
    main()
