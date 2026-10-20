
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from training.data_loader import (
    CLASS_NAMES,
    TEST_DIR,
    collect_images,
)
from xai.grad_cam import (
    MODEL_PATH,
    load_mri_image,
    generate_gradcam,
)
from tensorflow.keras.models import load_model

SEGMENTATION_ROOT = Path(
    r"D:\Downloads\dataset\brisc2025\segmentation_task"
)

IMAGE_DIR = SEGMENTATION_ROOT / "test" / "images"
MASK_DIR = SEGMENTATION_ROOT / "test" / "masks"

OUTPUT_DIR = Path("outputs/gradcam_mask_evaluation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# First experiment: two images from each class.
SAMPLES_PER_CLASS = 2

# These are experimental thresholds, not clinically validated values.
MASK_THRESHOLD = 128
HEATMAP_THRESHOLD = 0.5


def calculate_overlap(predicted_region, true_region):
    intersection = np.logical_and(
        predicted_region, true_region
    ).sum()

    predicted_area = predicted_region.sum()
    true_area = true_region.sum()

    dice = (
        2 * intersection / (predicted_area + true_area)
        if predicted_area + true_area > 0 else 1.0
    )

    union = np.logical_or(
        predicted_region, true_region
    ).sum()

    iou = intersection / union if union > 0 else 1.0

    return float(dice), float(iou)


def main():
    model = load_model(MODEL_PATH)

    test_paths, test_labels = collect_images(TEST_DIR)
    rng = np.random.default_rng(42)

    rows = []

    for class_index, class_name in enumerate(CLASS_NAMES):
        indices = np.where(test_labels == class_index)[0]
        selected = rng.choice(
            indices,
            size=min(SAMPLES_PER_CLASS, len(indices)),
            replace=False,
        )

        for index in selected:
            image_path = Path(test_paths[index])
            basename = image_path.stem

            mask_path = MASK_DIR / f"{basename}.png"
            segmentation_image_path = IMAGE_DIR / f"{basename}.jpg"

            if not mask_path.exists():
                print(f"Skipping; mask missing: {basename}")
                continue

            if not segmentation_image_path.exists():
                print(f"Skipping; segmentation image missing: {basename}")
                continue

            original, model_input = load_mri_image(image_path)

            # Explain the class predicted by the model.
            heatmap, predicted_index, probabilities = (
                generate_gradcam(model, model_input)
            )

            # Use the mask at its original 512x512 resolution.
            mask = np.array(
                Image.open(mask_path).convert("L")
            )
            true_region = mask >= MASK_THRESHOLD

            # Resize the heatmap to match the mask.
            heatmap_image = Image.fromarray(
                np.uint8(np.clip(heatmap, 0, 1) * 255)
            )
            heatmap_resized = np.array(
                heatmap_image.resize(
                    (mask.shape[1], mask.shape[0]),
                    Image.Resampling.BILINEAR,
                )
            ) / 255.0

            predicted_region = (
                heatmap_resized >= HEATMAP_THRESHOLD
            )

            dice, iou = calculate_overlap(
                predicted_region, true_region
            )

            rows.append({
                "image": basename,
                "actual_class": class_name,
                "predicted_class": CLASS_NAMES[predicted_index],
                "correct_prediction": (
                    predicted_index == test_labels[index]
                ),
                "confidence": float(
                    probabilities[predicted_index]
                ),
                "mask_threshold": MASK_THRESHOLD,
                "heatmap_threshold": HEATMAP_THRESHOLD,
                "mask_area_percent": float(
                    true_region.mean() * 100
                ),
                "heatmap_area_percent": float(
                    predicted_region.mean() * 100
                ),
                "dice": dice,
                "iou": iou,
            })

            print(
                f"{basename} | actual={class_name} | "
                f"predicted={CLASS_NAMES[predicted_index]} | "
                f"Dice={dice:.4f} | IoU={iou:.4f}"
            )

    output_csv = OUTPUT_DIR / "mask_overlap_results.csv"
    pd.DataFrame(rows).to_csv(output_csv, index=False)

    print(f"\nResults saved to: {output_csv.resolve()}")


if __name__ == "__main__":
    main()
