from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from training.data_loader import prepare_datasets, CLASS_NAMES

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "vgg16_brisc_best.keras"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
REPORT_DIR = PROJECT_ROOT / "reports"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    print("Loading test dataset...")
    _, _, test_ds = prepare_datasets()

    print("\nLoading trained model...")
    model = tf.keras.models.load_model(MODEL_PATH)

    print("\nEvaluating on the official test set...")
    test_loss, test_accuracy = model.evaluate(test_ds, verbose=1)

    true_labels = np.concatenate(
        [labels.numpy() for _, labels in test_ds]
    )

    probabilities = model.predict(test_ds, verbose=1)
    predicted_labels = np.argmax(probabilities, axis=1)

    report = classification_report(
        true_labels,
        predicted_labels,
        labels=list(range(len(CLASS_NAMES))),
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
    )

    matrix = confusion_matrix(
        true_labels,
        predicted_labels,
        labels=list(range(len(CLASS_NAMES))),
    )

    print("\n========== TEST RESULTS ==========")
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")

    print("\n========== CLASSIFICATION REPORT ==========")
    print(report)

    print("\n========== CONFUSION MATRIX ==========")
    print(matrix)

    results = {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "class_names": CLASS_NAMES,
        "confusion_matrix": matrix.tolist(),
        "classification_report": classification_report(
            true_labels,
            predicted_labels,
            labels=list(range(len(CLASS_NAMES))),
            target_names=CLASS_NAMES,
            output_dict=True,
            zero_division=0,
        ),
    }

    with (OUTPUT_DIR / "test_results.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(results, file, indent=2)

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=CLASS_NAMES,
    )
    display.plot(cmap="Blues", xticks_rotation=30)
    plt.title("VGG16 Classifier - Test Set Confusion Matrix")
    plt.tight_layout()
    plt.savefig(
        REPORT_DIR / "test_confusion_matrix.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()

    print(f"\nSaved results to: {OUTPUT_DIR / 'test_results.json'}")
    print(f"Saved confusion matrix to: {REPORT_DIR / 'test_confusion_matrix.png'}")


if __name__ == "__main__":
    main()
