
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from training.data_loader import CLASS_NAMES
from xai.grad_cam import (
    MODEL_PATH,
    load_mri_image,
    generate_gradcam,
)
from tensorflow.keras.models import load_model

ROOT = Path(
    r"D:\Downloads\dataset\brisc2025\segmentation_task"
)
IMAGE_DIR = ROOT / "test" / "images"
MASK_DIR = ROOT / "test" / "masks"

OUTPUT_DIR = Path("outputs/gradcam_mask_comparison")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLES = [
    "brisc2025_test_00023_gl_ax_t1",
    "brisc2025_test_00197_gl_sa_t1",
    "brisc2025_test_00388_me_ax_t1",
    "brisc2025_test_00387_me_ax_t1",
    "brisc2025_test_00729_pi_ax_t1",
    "brisc2025_test_00858_pi_co_t1",
]

MASK_THRESHOLD = 128

model = load_model(MODEL_PATH)

fig, axes = plt.subplots(
    len(SAMPLES), 3, figsize=(12, 4 * len(SAMPLES))
)

for row, basename in enumerate(SAMPLES):
    image_path = IMAGE_DIR / f"{basename}.jpg"
    mask_path = MASK_DIR / f"{basename}.png"

    if not image_path.exists() or not mask_path.exists():
        for ax in axes[row]:
            ax.axis("off")
        print(f"Missing segmentation pair: {basename}")
        continue

    original, model_input = load_mri_image(image_path)

    heatmap, predicted_index, probabilities = generate_gradcam(
        model, model_input
    )

    mask = np.array(Image.open(mask_path).convert("L"))
    tumor_region = mask >= MASK_THRESHOLD

    # Resize the heatmap to the original mask dimensions.
    heatmap_image = Image.fromarray(
        np.uint8(np.clip(heatmap, 0, 1) * 255)
    )
    heatmap_resized = np.array(
        heatmap_image.resize(
            (mask.shape[1], mask.shape[0]),
            Image.Resampling.BILINEAR,
        )
    ) / 255.0

    # Resize the MRI to match the mask for display.
    mri = np.array(
        Image.fromarray(original).resize(
            (mask.shape[1], mask.shape[0])
        )
    )

    axes[row, 0].imshow(mri)
    axes[row, 0].set_title(
        f"{basename}\nMRI"
    )

    axes[row, 1].imshow(mask, cmap="gray", vmin=0, vmax=255)
    axes[row, 1].set_title("Reference mask")

    axes[row, 2].imshow(mri)
    axes[row, 2].imshow(
        heatmap_resized, cmap="jet", alpha=0.45,
        vmin=0, vmax=1
    )
    axes[row, 2].contour(
        tumor_region.astype(float),
        levels=[0.5],
        colors=["white"],
        linewidths=1,
    )
    axes[row, 2].set_title(
        f"Grad-CAM: {CLASS_NAMES[predicted_index]} "
        f"({probabilities[predicted_index]:.1%})\n"
        "White outline = reference mask"
    )

    for ax in axes[row]:
        ax.axis("off")

plt.tight_layout()

output_path = OUTPUT_DIR / "comparison.png"
plt.savefig(output_path, dpi=140, bbox_inches="tight")
plt.close()

print(f"Saved comparison to: {output_path.resolve()}")
