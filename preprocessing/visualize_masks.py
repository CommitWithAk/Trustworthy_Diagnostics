
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

DATASET_ROOT = Path(
    r"D:\Downloads\dataset\brisc2025\segmentation_task"
)

IMAGE_DIR = DATASET_ROOT / "train" / "images"
MASK_DIR = DATASET_ROOT / "train" / "masks"

OUTPUT_DIR = Path("outputs/mask_inspection")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLES = [
    "brisc2025_train_00001_gl_ax_t1",
    "brisc2025_train_01967_me_co_t1",
    "brisc2025_train_05000_pi_sa_t1",
]

THRESHOLD = 128

fig, axes = plt.subplots(
    len(SAMPLES), 3, figsize=(12, 12)
)

for row, basename in enumerate(SAMPLES):
    image_path = IMAGE_DIR / f"{basename}.jpg"
    mask_path = MASK_DIR / f"{basename}.png"

    if not image_path.exists() or not mask_path.exists():
        print(f"Missing image or mask: {basename}")
        continue

    image = np.array(Image.open(image_path).convert("RGB"))
    mask = np.array(Image.open(mask_path).convert("L"))

    binary_mask = mask >= THRESHOLD

    # Overlay the selected mask region in red.
    overlay = image.copy()
    overlay[binary_mask] = (
        0.5 * overlay[binary_mask]
        + 0.5 * np.array([255, 0, 0])
    ).astype(np.uint8)

    axes[row, 0].imshow(image)
    axes[row, 0].set_title(f"{basename}\nMRI")

    axes[row, 1].imshow(mask, cmap="gray", vmin=0, vmax=255)
    axes[row, 1].set_title("Original mask")

    axes[row, 2].imshow(overlay)
    axes[row, 2].set_title(f"Mask overlay (>= {THRESHOLD})")

    for ax in axes[row]:
        ax.axis("off")

plt.tight_layout()

output_path = OUTPUT_DIR / "mask_visual_inspection.png"
plt.savefig(output_path, dpi=150, bbox_inches="tight")
plt.close()

print(f"Visualization saved to: {output_path.resolve()}")
