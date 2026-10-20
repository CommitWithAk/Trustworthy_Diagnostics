from pathlib import Path
import csv
import random

import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from training.data_loader import (
    TEST_DIR,
    CLASS_NAMES,
    collect_images,
)
from xai.grad_cam import (
    MODEL_PATH,
    load_mri_image,
    generate_gradcam,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "gradcam_evaluation"

SAMPLES_PER_CLASS = 2
RANDOM_SEED = 42


def save_visualization(original_image, heatmap, output_path):
    """Save the original MRI, heatmap, and overlay."""

    height, width = original_image.shape[:2]
    resized_heatmap = cv2.resize(heatmap, (width, height))

    heatmap_uint8 = np.uint8(255 * resized_heatmap)
    colored_heatmap = cv2.applyColorMap(
        heatmap_uint8, cv2.COLORMAP_JET
    )
    colored_heatmap = cv2.cvtColor(
        colored_heatmap, cv2.COLOR_BGR2RGB
    )

    overlay = cv2.addWeighted(
        original_image, 0.60,
        colored_heatmap, 0.40,
        0,
    )

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(original_image)
    axes[0].set_title("Original MRI")

    axes[1].imshow(resized_heatmap, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title("Grad-CAM")

    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")

    for ax in axes:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading test image paths...")
    paths, labels = collect_images(TEST_DIR)

    rng = random.Random(RANDOM_SEED)
    selected_indices = []

    # Select a reproducible random sample from every class.
    for class_index, class_name in enumerate(CLASS_NAMES):
        class_indices = [
            i for i, label in enumerate(labels)
            if label == class_index
        ]

        if len(class_indices) < SAMPLES_PER_CLASS:
            raise ValueError(
                f"Not enough test images for {class_name}"
            )

        chosen = rng.sample(class_indices, SAMPLES_PER_CLASS)
        selected_indices.extend(chosen)

    print(f"Loading model: {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH)

    results = []

    for count, index in enumerate(selected_indices, start=1):
        image_path = Path(paths[index])
        true_index = int(labels[index])
        true_class = CLASS_NAMES[true_index]

        print(f"\n[{count}/{len(selected_indices)}] {image_path.name}")

        original_image, model_input = load_mri_image(image_path)

        heatmap, explained_index, probabilities = generate_gradcam(
            model,
            model_input,
        )

        predicted_index = int(np.argmax(probabilities))
        predicted_class = CLASS_NAMES[predicted_index]
        confidence = float(probabilities[predicted_index])
        is_correct = predicted_index == true_index

        output_path = OUTPUT_DIR / (
            f"{image_path.stem}_gradcam.png"
        )

        save_visualization(
            original_image,
            heatmap,
            output_path,
        )

        results.append({
            "image": image_path.name,
            "image_path": str(image_path),
            "true_class": true_class,
            "predicted_class": predicted_class,
            "confidence": round(confidence, 6),
            "correct": is_correct,
            "explained_class": CLASS_NAMES[explained_index],
            "gradcam_path": str(output_path),
        })

        print(f"Actual: {true_class}")
        print(f"Predicted: {predicted_class}")
        print(f"Confidence: {confidence:.2%}")
        print(f"Correct: {is_correct}")
        print(f"Saved: {output_path}")

    csv_path = OUTPUT_DIR / "gradcam_evaluation.csv"

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=results[0].keys(),
        )
        writer.writeheader()
        writer.writerows(results)

    correct_count = sum(row["correct"] for row in results)
    total_count = len(results)

    print("\n========== SAMPLE SUMMARY ==========")
    print(f"Images evaluated: {total_count}")
    print(f"Correct predictions: {correct_count}")
    print(f"Incorrect predictions: {total_count - correct_count}")
    print(f"Sample accuracy: {correct_count / total_count:.2%}")
    print(f"CSV report: {csv_path}")
    print(f"Visualizations: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
