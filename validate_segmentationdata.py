
from pathlib import Path
from collections import Counter
from PIL import Image
import numpy as np

ROOT = Path(r"D:\Downloads\dataset\brisc2025")

splits = {
    "train": ROOT / "segmentation_task" / "train",
    "test": ROOT / "segmentation_task" / "test",
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

for split_name, split_path in splits.items():
    image_dir = split_path / "images"
    mask_dir = split_path / "masks"

    print(f"\n{'=' * 55}")
    print(f"Checking segmentation {split_name}")
    print(f"{'=' * 55}")

    images = {
        p.stem: p for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    }
    masks = {
        p.stem: p for p in mask_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    }

    missing_masks = sorted(set(images) - set(masks))
    missing_images = sorted(set(masks) - set(images))

    print("Images:", len(images))
    print("Masks:", len(masks))
    print("Images without matching masks:", len(missing_masks))
    print("Masks without matching images:", len(missing_images))

    if missing_masks:
        print("Example missing masks:", missing_masks[:5])
    if missing_images:
        print("Example missing images:", missing_images[:5])

    checked = 0
    unreadable = []
    dimension_mismatches = []
    mask_value_counts = Counter()
    empty_masks = 0

    for stem in sorted(set(images) & set(masks)):
        image_path = images[stem]
        mask_path = masks[stem]

        try:
            with Image.open(image_path) as image:
                image.load()
                image_size = image.size

            with Image.open(mask_path) as mask:
                mask.load()
                mask_array = np.asarray(mask)
                mask_size = mask.size

            if image_size != mask_size:
                dimension_mismatches.append(
                    (stem, image_size, mask_size)
                )

            values = np.unique(mask_array)
            mask_value_counts.update(
                str(value) for value in values.tolist()
            )

            if np.count_nonzero(mask_array) == 0:
                empty_masks += 1

            checked += 1

        except Exception as exc:
            unreadable.append((stem, str(exc)))

    print("Pairs checked:", checked)
    print("Unreadable pairs:", len(unreadable))
    print("Image-mask dimension mismatches:", len(dimension_mismatches))
    print("All-zero masks:", empty_masks)
    print("Mask pixel values encountered:", sorted(mask_value_counts.keys()))

    if unreadable:
        print("Example unreadable pairs:", unreadable[:3])
    if dimension_mismatches:
        print("Example dimension mismatches:", dimension_mismatches[:3])
