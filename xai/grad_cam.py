from pathlib import Path
import argparse

import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.applications.vgg16 import preprocess_input
from training.data_loader import (
    TEST_DIR,
    CLASS_NAMES,
    IMAGE_SIZE,
    collect_images,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "vgg16_brisc_best.keras"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_mri_image(image_path):
    """Load an MRI image and prepare it for VGG16."""
    image_bytes = tf.io.read_file(str(image_path))
    image = tf.io.decode_image(
        image_bytes,
        channels=3,
        expand_animations=False,
    )
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.cast(image, tf.float32)

   
    original_image = image.numpy().astype(np.uint8)

  
    model_input = preprocess_input(
        np.expand_dims(image.numpy(), axis=0)
    )

    return original_image, tf.convert_to_tensor(model_input)


def generate_gradcam(model, model_input, class_index=None):
    """Generate a Grad-CAM heatmap for the selected class."""

    # VGG16 is nested inside our trained classification model.
    base_model = model.get_layer("vgg16")
    last_conv_layer = base_model.get_layer("block5_conv3")

    # Return convolutional activations and final VGG16 features
    # from the same forward pass.
    feature_extractor = tf.keras.Model(
        inputs=base_model.inputs,
        outputs=[
            last_conv_layer.output,
            base_model.output,
        ],
    )

    gap_layer = model.get_layer("global_average_pooling")
    dropout_layer = model.get_layer("dropout")
    classifier_layer = model.get_layer("class_probabilities")

    with tf.GradientTape() as tape:
        conv_outputs, features = feature_extractor(
            model_input, training=False
        )

        # Reproduce the trained classification head.
        x = gap_layer(features)
        x = dropout_layer(x, training=False)
        predictions = classifier_layer(x)

        if class_index is None:
            class_index = int(
                tf.argmax(predictions[0]).numpy()
            )

        class_score = predictions[:, class_index]

    gradients = tape.gradient(class_score, conv_outputs)

    if gradients is None:
        raise RuntimeError(
            "Gradients could not be calculated. "
            "Check the model's convolutional layer."
        )

    # Average gradients across the spatial dimensions.
    channel_weights = tf.reduce_mean(
        gradients, axis=(1, 2)
    )

    # Weight each feature map by its importance.
    heatmap = tf.reduce_sum(
        conv_outputs * channel_weights[:, tf.newaxis, tf.newaxis, :],
        axis=-1,
    )

    heatmap = tf.nn.relu(heatmap[0])
    max_value = tf.reduce_max(heatmap)

    heatmap = tf.math.divide_no_nan(heatmap, max_value)
    heatmap = heatmap.numpy()

    return heatmap, class_index, predictions.numpy()[0]


def save_gradcam_overlay(original_image, heatmap, output_path):
    """Create and save an MRI image with its Grad-CAM overlay."""

    height, width = original_image.shape[:2]

    resized_heatmap = cv2.resize(
        heatmap, (width, height)
    )
    heatmap_uint8 = np.uint8(
        255 * resized_heatmap
    )

    # OpenCV creates a BGR heatmap; convert it to RGB.
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

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(original_image)
    axes[0].set_title("Original MRI")

    axes[1].imshow(resized_heatmap, cmap="jet")
    axes[1].set_title("Grad-CAM Heatmap")

    axes[2].imshow(overlay)
    axes[2].set_title("MRI + Grad-CAM")

    for axis in axes:
        axis.axis("off")

    plt.tight_layout()
    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )
    plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Generate a Grad-CAM explanation for an MRI image."
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to an MRI image. Defaults to the first test image.",
    )
    parser.add_argument(
        "--class-index",
        type=int,
        default=None,
        choices=range(len(CLASS_NAMES)),
        help="Optional class index to explain instead of the predicted class.",
    )
    args = parser.parse_args()

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}"
        )

    if args.image:
        image_path = Path(args.image)
        if not image_path.is_file():
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )
    else:
        test_paths, _ = collect_images(TEST_DIR)
        image_path = Path(test_paths[0])

    print(f"Selected image: {image_path}")

    original_image, model_input = load_mri_image(image_path)

    print("Loading trained model...")
    model = tf.keras.models.load_model(MODEL_PATH)

    heatmap, predicted_index, probabilities = generate_gradcam(
        model,
        model_input,
        class_index=args.class_index,
    )

    # Report the model's actual top prediction separately
    # from the class chosen for explanation.
    top_index = int(np.argmax(probabilities))

    print("\n========== PREDICTION ==========")
    print(f"Predicted class: {CLASS_NAMES[top_index]}")
    print(f"Prediction confidence: {probabilities[top_index]:.2%}")
    print(f"Class being explained: {CLASS_NAMES[predicted_index]}")

    print("\nClass probabilities:")
    for name, probability in zip(CLASS_NAMES, probabilities):
        print(f"{name}: {probability:.2%}")

    output_path = OUTPUT_DIR / (
        f"gradcam_{image_path.stem}_{CLASS_NAMES[predicted_index]}.png"
    )

    save_gradcam_overlay(
        original_image,
        heatmap,
        output_path,
    )

    print(f"\nGrad-CAM image saved to: {output_path}")


if __name__ == "__main__":
    main()
