
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.vgg16 import preprocess_input

DATASET_ROOT = Path(
    r"D:\Downloads\dataset\brisc2025\classification_task"
)

TRAIN_DIR = DATASET_ROOT / "train"
TEST_DIR = DATASET_ROOT / "test"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
VALIDATION_SIZE = 0.20
SEED = 42

CLASS_NAMES = [
    "glioma",
    "meningioma",
    "no_tumor",
    "pituitary",
]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def collect_images(folder):
    """Collect image paths and labels from class subfolders."""
    paths = []
    labels = []

    for label, class_name in enumerate(CLASS_NAMES):
        class_dir = folder / class_name

        if not class_dir.is_dir():
            raise FileNotFoundError(
                f"Missing class directory: {class_dir}"
            )

        class_paths = sorted(
            p for p in class_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )

        paths.extend(str(p) for p in class_paths)
        labels.extend([label] * len(class_paths))

    if not paths:
        raise ValueError(f"No images found in {folder}")

    return np.array(paths), np.array(labels, dtype=np.int32)


def load_image(path, label):
    """Load and preprocess one image for VGG16."""
    image_bytes = tf.io.read_file(path)
    image = tf.io.decode_image(
        image_bytes,
        channels=3,
        expand_animations=False,
    )
    image.set_shape([None, None, 3])

    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.cast(image, tf.float32)

    image = preprocess_input(image)

    return image, label


def create_dataset(paths, labels, training=False):
    """Build an efficient tf.data input pipeline."""
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))

    if training:
        dataset = dataset.shuffle(
            buffer_size=len(paths),
            seed=SEED,
            reshuffle_each_iteration=True,
        )

    dataset = dataset.map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


def prepare_datasets():
    """Create training, validation, and official test datasets."""
    train_paths, train_labels = collect_images(TRAIN_DIR)
    test_paths, test_labels = collect_images(TEST_DIR)

    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_paths,
        train_labels,
        test_size=VALIDATION_SIZE,
        random_state=SEED,
        stratify=train_labels,
    )

    train_ds = create_dataset(
        train_paths, train_labels, training=True
    )
    val_ds = create_dataset(
        val_paths, val_labels
    )
    test_ds = create_dataset(
        test_paths, test_labels
    )

    print("Class mapping:", dict(enumerate(CLASS_NAMES)))
    print(f"Training images:   {len(train_paths)}")
    print(f"Validation images: {len(val_paths)}")
    print(f"Test images:       {len(test_paths)}")

    print("\nTraining distribution:")
    for label, name in enumerate(CLASS_NAMES):
        print(f"{name}: {np.sum(train_labels == label)}")

    print("\nValidation distribution:")
    for label, name in enumerate(CLASS_NAMES):
        print(f"{name}: {np.sum(val_labels == label)}")

    return train_ds, val_ds, test_ds


if __name__ == "__main__":
    train_ds, val_ds, test_ds = prepare_datasets()

    images, labels = next(iter(train_ds))

    print("\nFirst training batch:")
    print("Image tensor shape:", images.shape)
    print("Label tensor shape:", labels.shape)
    print("Image tensor dtype:", images.dtype)
    print("Labels in batch:", labels.numpy())
