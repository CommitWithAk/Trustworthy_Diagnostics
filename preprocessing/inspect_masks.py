
from pathlib import Path
import numpy as np
from PIL import Image

# Segmentation dataset paths
DATASET_ROOT = Path(
    r"D:\Downloads\dataset\brisc2025\segmentation_task"
)

IMAGE_DIR = DATASET_ROOT / "train" / "images"
MASK_DIR = DATASET_ROOT / "train" / "masks"

# Inspect three sample masks
SAMPLE_MASKS = [
    "brisc2025_train_00001_gl_ax_t1.png",
    "brisc2025_train_01967_me_co_t1.png",
    "brisc2025_train_05000_pi_sa_t1.png",
]

for filename in SAMPLE_MASKS:
    mask_path = MASK_DIR / filename

    if not mask_path.exists():
        print(f"\nMask not found: {mask_path}")
        continue

    mask = np.array(
        Image.open(mask_path).convert("L")
    )

    print(f"\nMask: {filename}")
    print("Shape:", mask.shape)
    print("Data type:", mask.dtype)
    print("Minimum pixel value:", mask.min())
    print("Maximum pixel value:", mask.max())
    print("Unique pixel values:", np.unique(mask)[:30])

    for threshold in [0, 10, 128, 250, 255]:
        foreground = mask >= threshold if threshold > 0 else mask > 0
        percentage = foreground.mean() * 100
        print(
            f"Pixels selected at threshold {threshold}: "
            f"{percentage:.3f}%"
        )

    # Check whether the paired MRI image exists
    image_path = IMAGE_DIR / filename.replace(".png", ".jpg")
    print("Paired MRI exists:", image_path.exists())
