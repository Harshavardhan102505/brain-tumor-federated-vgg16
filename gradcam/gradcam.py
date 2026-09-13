import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "federated_vgg16_round5.keras"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "gradcam"
    / "outputs"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

IMG_SIZE = 224
THRESHOLD = 0.50

# Last convolutional layer inside VGG16
LAST_CONV_LAYER = "block5_conv3"


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("FEDERATED VGG16 - GRAD-CAM")
print("=" * 70)


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading federated model...")

if not MODEL_PATH.exists():

    print("\nERROR: Model not found:")
    print(MODEL_PATH)

    sys.exit(1)


model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

print(
    "Model loaded successfully:"
)

print(
    MODEL_PATH.name
)


# ============================================================
# MODEL STRUCTURE
# ============================================================

print("\nOuter model:")

for i, layer in enumerate(model.layers):

    print(
        i,
        layer.name,
        layer.__class__.__name__,
        getattr(
            layer,
            "output_shape",
            ""
        )
    )


# ============================================================
# GET VGG16 SUBMODEL
# ============================================================

vgg16 = model.layers[0]

print("\nVGG16 submodel found:")
print(vgg16.name)


# ============================================================
# GET LAST CONVOLUTIONAL LAYER
# ============================================================

try:

    last_conv_layer = vgg16.get_layer(
        LAST_CONV_LAYER
    )

except ValueError:

    print(
        "\nERROR: Could not find:",
        LAST_CONV_LAYER
    )

    print(
        "\nAvailable Conv2D layers:"
    )

    for layer in vgg16.layers:

        if isinstance(
            layer,
            tf.keras.layers.Conv2D
        ):

            print(
                layer.name,
                layer.output_shape
            )

    sys.exit(1)


print(
    "\nGrad-CAM layer:"
)

print(
    last_conv_layer.name
)

print(
    "Output:",
    last_conv_layer.output_shape
)


# ============================================================
# BUILD GRAD-CAM FEATURE MODEL
# ============================================================

# IMPORTANT:
#
# VGG16 is a nested model.
#
# Therefore we use the VGG16 model's own input:
#
# vgg16.input
#
# instead of:
#
# model.inputs
#
# This prevents the "Graph disconnected" error.

gradcam_feature_model = tf.keras.models.Model(
    inputs=vgg16.input,
    outputs=last_conv_layer.output
)


# ============================================================
# BUILD CLASSIFIER MODEL
# ============================================================

# The outer model after VGG16 contains:
#
# GlobalAveragePooling2D
# Dense(128)
# Dropout
# Dense(1)
#
# We reproduce this forward pass for Grad-CAM.

global_pool = model.layers[1]
dense_layer = model.layers[2]
dropout_layer = model.layers[3]
final_layer = model.layers[4]


# ============================================================
# LOAD IMAGE / NPY
# ============================================================

def load_input_image(image_path):

    image_path = Path(
        image_path
    )

    suffix = (
        image_path.suffix.lower()
    )

    # --------------------------------------------------------
    # NPY INPUT
    # --------------------------------------------------------

    if suffix == ".npy":

        print(
            "\nLoading NumPy MRI:"
        )

        print(
            image_path.name
        )

        image_array = np.load(
            image_path
        )

        print(
            "Original NPY shape:",
            image_array.shape
        )

        print(
            "Original dtype:",
            image_array.dtype
        )

        print(
            "Original range:",
            float(image_array.min()),
            "to",
            float(image_array.max())
        )

        # NPY files from your project are already:
        #
        # 224 x 224 x 3
        # float32
        # values approximately 0-1
        #
        # So don't divide by 255 again.

        if image_array.ndim == 2:

            image_array = np.stack(
                [
                    image_array,
                    image_array,
                    image_array
                ],
                axis=-1
            )

        if image_array.ndim != 3:

            raise ValueError(
                "Unsupported NPY shape: "
                + str(image_array.shape)
            )

        # Ensure RGB

        if image_array.shape[-1] == 1:

            image_array = np.repeat(
                image_array,
                3,
                axis=-1
            )

        elif image_array.shape[-1] != 3:

            raise ValueError(
                "Expected 3 channels, "
                f"got {image_array.shape[-1]}"
            )

        # Convert to float32

        image_array = image_array.astype(
            np.float32
        )

        # ----------------------------------------------------
        # Handle possible 0-255 NPY
        # ----------------------------------------------------

        if image_array.max() > 1.0:

            image_array = (
                image_array / 255.0
            )

        # Clip

        image_array = np.clip(
            image_array,
            0.0,
            1.0
        )

        # Convert to PIL for visualization

        display_array = (
            image_array * 255
        ).astype(
            np.uint8
        )

        display_image = Image.fromarray(
            display_array
        ).convert(
            "RGB"
        )

        # Resize if necessary

        if display_image.size != (
            IMG_SIZE,
            IMG_SIZE
        ):

            display_image = (
                display_image.resize(
                    (IMG_SIZE, IMG_SIZE)
                )
            )

            image_array = (
                np.array(
                    display_image,
                    dtype=np.float32
                )
                / 255.0
            )

        return (
            display_image,
            np.expand_dims(
                image_array,
                axis=0
            )
        )

    # --------------------------------------------------------
    # JPG / JPEG / PNG INPUT
    # --------------------------------------------------------

    elif suffix in [
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp"
    ]:

        print(
            "\nLoading image:"
        )

        print(
            image_path.name
        )

        image = Image.open(
            image_path
        ).convert(
            "RGB"
        )

        print(
            "Original size:",
            image.size
        )

        image = image.resize(
            (
                IMG_SIZE,
                IMG_SIZE
            )
        )

        image_array = np.array(
            image,
            dtype=np.float32
        )

        image_array = (
            image_array / 255.0
        )

        image_array = np.clip(
            image_array,
            0.0,
            1.0
        )

        return (
            image,
            np.expand_dims(
                image_array,
                axis=0
            )
        )

    else:

        raise ValueError(
            "Unsupported file format.\n"
            "Use JPG, JPEG, PNG or NPY."
        )


# ============================================================
# GRAD-CAM
# ============================================================

def generate_gradcam(
    input_array
):

    # Convert input to TensorFlow tensor

    input_tensor = tf.convert_to_tensor(
        input_array,
        dtype=tf.float32
    )

    # --------------------------------------------------------
    # Gradient calculation
    # --------------------------------------------------------

    with tf.GradientTape() as tape:

        # Get convolutional feature maps

        conv_outputs = (
            gradcam_feature_model(
                input_tensor
            )
        )

        # Continue through classifier
        #
        # IMPORTANT:
        # We use the same layers from the
        # trained federated model.

        x = global_pool(
            conv_outputs
        )

        x = dense_layer(
            x
        )

        # Dropout is disabled during inference

        x = dropout_layer(
            x,
            training=False
        )

        predictions = final_layer(
            x
        )

        tumor_probability = (
            predictions[:, 0]
        )

    # --------------------------------------------------------
    # Gradients
    # --------------------------------------------------------

    gradients = tape.gradient(
        tumor_probability,
        conv_outputs
    )

    if gradients is None:

        raise RuntimeError(
            "Gradients could not be calculated."
        )

    # --------------------------------------------------------
    # Global average pooling of gradients
    # --------------------------------------------------------

    pooled_gradients = tf.reduce_mean(
        gradients,
        axis=(1, 2)
    )

    # --------------------------------------------------------
    # Remove batch dimension
    # --------------------------------------------------------

    conv_outputs = conv_outputs[0]

    pooled_gradients = (
        pooled_gradients[0]
    )

    # --------------------------------------------------------
    # Weighted feature maps
    # --------------------------------------------------------

    weighted_features = (
        conv_outputs
        *
        pooled_gradients
    )

    heatmap = tf.reduce_sum(
        weighted_features,
        axis=-1
    )

    # --------------------------------------------------------
    # ReLU
    # --------------------------------------------------------

    heatmap = tf.maximum(
        heatmap,
        0
    )

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    max_value = tf.reduce_max(
        heatmap
    )

    if float(max_value.numpy()) > 0:

        heatmap = (
            heatmap / max_value
        )

    heatmap = heatmap.numpy()

    probability = float(
        tumor_probability.numpy()[0]
    )

    return (
        heatmap,
        probability
    )


# ============================================================
# CREATE HEATMAP
# ============================================================

def create_heatmap_image(
    heatmap,
    size
):

    heatmap_uint8 = (
        heatmap * 255
    ).astype(
        np.uint8
    )

    heatmap_image = Image.fromarray(
        heatmap_uint8
    )

    heatmap_image = (
        heatmap_image.resize(
            size,
            Image.Resampling.BILINEAR
        )
    )

    return heatmap_image


# ============================================================
# CREATE OVERLAY
# ============================================================

def create_overlay(
    original_image,
    heatmap,
    alpha=0.45
):

    heatmap_image = (
        create_heatmap_image(
            heatmap,
            original_image.size
        )
    )

    heatmap_array = np.array(
        heatmap_image
    )

    # Jet colormap

    cmap = plt.get_cmap(
        "jet"
    )

    colored_heatmap = cmap(
        heatmap_array / 255.0
    )[:, :, :3]

    colored_heatmap = (
        colored_heatmap * 255
    ).astype(
        np.uint8
    )

    original_array = np.array(
        original_image
    ).astype(
        np.float32
    )

    colored_heatmap = (
        colored_heatmap.astype(
            np.float32
        )
    )

    overlay = (
        (1 - alpha)
        * original_array
        +
        alpha
        * colored_heatmap
    )

    overlay = np.clip(
        overlay,
        0,
        255
    ).astype(
        np.uint8
    )

    return Image.fromarray(
        overlay
    )


# ============================================================
# PROCESS IMAGE
# ============================================================

def process_image(
    image_path
):

    image_path = Path(
        image_path
    )

    if not image_path.exists():

        print(
            "\nERROR: File does not exist:"
        )

        print(
            image_path
        )

        return

    print("\n" + "=" * 70)
    print("PROCESSING")
    print("=" * 70)

    print(
        "\nFile:",
        image_path
    )

    # --------------------------------------------------------
    # Load image
    # --------------------------------------------------------

    try:

        original_image, input_array = (
            load_input_image(
                image_path
            )
        )

    except Exception as e:

        print(
            "\nERROR while loading image:"
        )

        print(e)

        return

    print(
        "\nModel input shape:",
        input_array.shape
    )

    # --------------------------------------------------------
    # Prediction + Grad-CAM
    # --------------------------------------------------------

    print(
        "\nGenerating prediction..."
    )

    try:

        heatmap, probability = (
            generate_gradcam(
                input_array
            )
        )

    except Exception as e:

        print(
            "\nERROR during Grad-CAM:"
        )

        print(e)

        return

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    if probability >= THRESHOLD:

        predicted_class = "Tumor"

        confidence = (
            probability * 100
        )

    else:

        predicted_class = "No Tumor"

        confidence = (
            (1.0 - probability)
            * 100
        )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print("\n" + "=" * 50)

    print(
        "PREDICTION RESULTS"
    )

    print("=" * 50)

    print(
        "Predicted Class:",
        predicted_class
    )

    print(
        "Tumor Probability:",
        f"{probability:.4f}"
    )

    print(
        "No Tumor Probability:",
        f"{1.0 - probability:.4f}"
    )

    print(
        "Confidence:",
        f"{confidence:.2f}%"
    )

    print(
        "Decision Threshold:",
        THRESHOLD
    )

    print("=" * 50)

    # --------------------------------------------------------
    # Output names
    # --------------------------------------------------------

    stem = image_path.stem

    overlay_path = (
        OUTPUT_DIR
        /
        f"{stem}_gradcam.jpg"
    )

    heatmap_path = (
        OUTPUT_DIR
        /
        f"{stem}_heatmap.jpg"
    )

    comparison_path = (
        OUTPUT_DIR
        /
        f"{stem}_comparison.jpg"
    )

    # --------------------------------------------------------
    # Create overlay
    # --------------------------------------------------------

    overlay = create_overlay(
        original_image,
        heatmap,
        alpha=0.45
    )

    overlay.save(
        overlay_path,
        quality=95
    )

    # --------------------------------------------------------
    # Save heatmap
    # --------------------------------------------------------

    heatmap_image = (
        create_heatmap_image(
            heatmap,
            original_image.size
        )
    )

    heatmap_image.save(
        heatmap_path,
        quality=95
    )

    # --------------------------------------------------------
    # Comparison figure
    # --------------------------------------------------------

    plt.figure(
        figsize=(12, 5)
    )

    plt.subplot(
        1,
        2,
        1
    )

    plt.imshow(
        original_image
    )

    plt.title(
        "Original MRI"
    )

    plt.axis(
        "off"
    )

    plt.subplot(
        1,
        2,
        2
    )

    plt.imshow(
        overlay
    )

    plt.title(
        "Grad-CAM\n"
        f"{predicted_class} - "
        f"{confidence:.2f}%"
    )

    plt.axis(
        "off"
    )

    plt.tight_layout()

    plt.savefig(
        comparison_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    # --------------------------------------------------------
    # Save numerical heatmap
    # --------------------------------------------------------

    numerical_heatmap_path = (
        OUTPUT_DIR
        /
        f"{stem}_heatmap.npy"
    )

    np.save(
        numerical_heatmap_path,
        heatmap
    )

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    print("\nOUTPUT FILES")
    print("-" * 50)

    print(
        "Grad-CAM:",
        overlay_path
    )

    print(
        "Heatmap:",
        heatmap_path
    )

    print(
        "Comparison:",
        comparison_path
    )

    print(
        "Heatmap NumPy:",
        numerical_heatmap_path
    )

    print("\n" + "=" * 70)
    print("GRAD-CAM COMPLETE")
    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    if len(sys.argv) < 2:

        print(
            "\nPlease provide an MRI image."
        )

        print(
            "\nExamples:"
        )

        print(
            r'python gradcam\gradcam.py "data\federated_clients\client_1\Tumor\Te-aug-me_8.npy"'
        )

        print(
            r'python gradcam\gradcam.py "image.jpg"'
        )

        sys.exit(1)

    image_path = sys.argv[1]

    process_image(
        image_path
    )