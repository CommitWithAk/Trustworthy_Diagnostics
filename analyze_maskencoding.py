
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(r"D:\Downloads\dataset\brisc2025")
IMAGE_DIR = ROOT / "segmentation_task" / "train" / "images"
MASK_DIR = ROOT / "segmentation_task" / "train" / "masks"

names = [
    "brisc2025_train_00001_gl_ax_t1",
    "brisc2025_train_01967_me_co_t1",
    "brisc2025_train_05000_pi_sa_t1",
]

for name in names:
    image_path = IMAGE_DIR / f"{name}.jpg"
    mask_path = MASK_DIR / f"{name}.png"

    with Image.open(image_path) as im:
        image = np.asarray(im.convert("RGB"))

    with Image.open(mask_path) as m:
        mask = np.asarray(m.convert("L"))

    binary_1 = mask > 0
    binary_2 = mask >= 128
    binary_3 = mask == 255

    print(f"\n{name}")
    for label, binary in [
        ("mask > 0", binary_1),
        ("mask >= 128", binary_2),
        ("mask == 255", binary_3),
    ]:
        print(
            f"{label}: {binary.sum()} pixels "
            f"({binary.mean() * 100:.3f}% of image)"
        )

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(image)
    axes[0].set_title("MRI")

    axes[1].imshow(mask, cmap="gray", vmin=0, vmax=255)
    axes[1].set_title("Original mask")

    axes[2].imshow(binary_2, cmap="gray")
    axes[2].set_title("Candidate: >= 128")

    overlay = image.copy()
    region = binary_2
    overlay[region] = (
        0.5 * overlay[region] + 0.5 * np.array([255, 0, 0])
    ).astype(np.uint8)

    axes[3].imshow(overlay)
    axes[3].set_title("Candidate overlay")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.show()
