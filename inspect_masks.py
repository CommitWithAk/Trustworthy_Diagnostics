
from pathlib import Path
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(r"D:\Downloads\dataset\brisc2025")
IMAGE_DIR = ROOT / "segmentation_task" / "train" / "images"
MASK_DIR = ROOT / "segmentation_task" / "train" / "masks"

image_files = sorted(IMAGE_DIR.glob("*.jpg"))

if not image_files:
    raise FileNotFoundError(f"No JPG images found in {IMAGE_DIR}")

indices = np.linspace(
    0, len(image_files) - 1, min(3, len(image_files)), dtype=int
)

fig, axes = plt.subplots(len(indices), 2, figsize=(10, 4 * len(indices)))
axes = np.atleast_2d(axes)

for row, index in enumerate(indices):
    image_path = image_files[index]
    mask_path = MASK_DIR / f"{image_path.stem}.png"

    if not mask_path.exists():
        print("Missing mask:", mask_path.name)
        continue

    with Image.open(image_path) as img:
        image = img.convert("RGB").copy()

    with Image.open(mask_path) as mask:
        mask_array = np.array(mask).copy()

    counts = Counter(mask_array.ravel().tolist())

    print(f"\nImage: {image_path.name}")
    print(f"Mask:  {mask_path.name}")
    print(f"Mask shape: {mask_array.shape}")
    print(f"Mask values and pixel counts: {sorted(counts.items())}")

    axes[row, 0].imshow(image)
    axes[row, 0].set_title(f"MRI: {image_path.name}")
    axes[row, 0].axis("off")

    axes[row, 1].imshow(mask_array, cmap="gray", vmin=0, vmax=255)
    axes[row, 1].set_title("Original mask")
    axes[row, 1].axis("off")

plt.tight_layout()
plt.show()
