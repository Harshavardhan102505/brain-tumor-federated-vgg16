import os
from pathlib import Path

import numpy as np
import tensorflow as tf

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for
)

from werkzeug.utils import secure_filename
from PIL import Image

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "federated_vgg16_round5.h5"
)

UPLOAD_FOLDER = (
    PROJECT_ROOT
    / "flask_app"
    / "static"
    / "uploads"
)

OUTPUT_FOLDER = (
    PROJECT_ROOT
    / "flask_app"
    / "outputs"
)

TEMPLATE_FOLDER = (
    PROJECT_ROOT
    / "flask_app"
    / "templates"
)


# ============================================================
# CREATE DIRECTORIES
# ============================================================

UPLOAD_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# FLASK APPLICATION
# ============================================================

app = Flask(
    __name__,
    template_folder=str(TEMPLATE_FOLDER)
)

app.config["UPLOAD_FOLDER"] = str(
    UPLOAD_FOLDER
)

app.config["MAX_CONTENT_LENGTH"] = (
    200 * 1024 * 1024
)


# ============================================================
# ALLOWED FILE TYPES
# ============================================================

ALLOWED_EXTENSIONS = {
    "jpg",
    "jpeg",
    "png"
}

IMG_SIZE = 224

THRESHOLD = 0.50


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 70)
print("BRAIN TUMOR DETECTION - FLASK APPLICATION")
print("=" * 70)

print()
print("Loading federated VGG16 model...")

if not MODEL_PATH.exists():

    raise FileNotFoundError(
        f"Model not found:\n{MODEL_PATH}"
    )

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

print("Model loaded successfully.")
print(
    f"Model: {MODEL_PATH.name}"
)


# ============================================================
# GET VGG16 SUBMODEL
# ============================================================

vgg16 = model.layers[0]

last_conv_layer = vgg16.get_layer(
    "block5_conv3"
)

print(
    f"Grad-CAM layer: {last_conv_layer.name}"
)


# ============================================================
# GRAD-CAM FEATURE MODEL
# ============================================================

gradcam_feature_model = tf.keras.models.Model(
    inputs=vgg16.input,
    outputs=last_conv_layer.output
)


# ============================================================
# CLASSIFIER LAYERS
# ============================================================

global_pool = model.layers[1]

dense_layer = model.layers[2]

dropout_layer = model.layers[3]

final_layer = model.layers[4]


# ============================================================
# CHECK FILE EXTENSION
# ============================================================

def allowed_file(filename):

    return (
        "." in filename
        and
        filename.rsplit(
            ".",
            1
        )[1].lower()
        in ALLOWED_EXTENSIONS
    )


# ============================================================
# PREPROCESS IMAGE
# ============================================================

def preprocess_image(image):

    image = image.convert("RGB")

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

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    return image_array


# ============================================================
# GENERATE GRAD-CAM
# ============================================================

def generate_gradcam(
    input_array
):

    input_tensor = tf.convert_to_tensor(
        input_array,
        dtype=tf.float32
    )

    with tf.GradientTape() as tape:

        conv_outputs = (
            gradcam_feature_model(
                input_tensor
            )
        )

        x = global_pool(
            conv_outputs
        )

        x = dense_layer(
            x
        )

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

    gradients = tape.gradient(
        tumor_probability,
        conv_outputs
    )

    if gradients is None:

        raise RuntimeError(
            "Unable to calculate Grad-CAM gradients."
        )

    pooled_gradients = tf.reduce_mean(
        gradients,
        axis=(1, 2)
    )

    conv_outputs = conv_outputs[0]

    pooled_gradients = (
        pooled_gradients[0]
    )

    weighted_features = (
        conv_outputs
        *
        pooled_gradients
    )

    heatmap = tf.reduce_sum(
        weighted_features,
        axis=-1
    )

    heatmap = tf.maximum(
        heatmap,
        0
    )

    maximum = tf.reduce_max(
        heatmap
    )

    if float(maximum.numpy()) > 0:

        heatmap = (
            heatmap / maximum
        )

    return (
        heatmap.numpy(),
        float(
            tumor_probability.numpy()[0]
        )
    )


# ============================================================
# CREATE GRAD-CAM OVERLAY
# ============================================================

def create_gradcam_overlay(
    image,
    heatmap
):

    heatmap_uint8 = (
        heatmap * 255
    ).astype(
        np.uint8
    )

    heatmap_image = Image.fromarray(
        heatmap_uint8
    )

    heatmap_image = heatmap_image.resize(
        image.size,
        Image.Resampling.BILINEAR
    )

    heatmap_array = np.array(
        heatmap_image
    )

    cmap = plt.get_cmap(
        "jet"
    )

    colored_heatmap = cmap(
        heatmap_array / 255.0
    )

    colored_heatmap = (
        colored_heatmap[:, :, :3]
        * 255
    ).astype(
        np.uint8
    )

    original_array = np.array(
        image
    ).astype(
        np.float32
    )

    colored_heatmap = (
        colored_heatmap.astype(
            np.float32
        )
    )

    overlay = (
        0.55 * original_array
        +
        0.45 * colored_heatmap
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
# SAVE HEATMAP
# ============================================================

def save_heatmap(
    heatmap,
    output_path
):

    plt.figure(
        figsize=(6, 6)
    )

    plt.imshow(
        heatmap,
        cmap="jet"
    )

    plt.axis("off")

    plt.tight_layout(
        pad=0
    )

    plt.savefig(
        output_path,
        bbox_inches="tight",
        pad_inches=0
    )

    plt.close()


# ============================================================
# HOME PAGE
# ============================================================

@app.route(
    "/",
    methods=["GET"]
)
def home():

    return render_template(
        "index.html"
    )


# ============================================================
# PREDICTION
# ============================================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    # --------------------------------------------------------
    # CHECK FILE
    # --------------------------------------------------------

    if "file" not in request.files:

        return redirect(
            url_for("home")
        )

    file = request.files["file"]

    if file.filename == "":

        return redirect(
            url_for("home")
        )

    if not allowed_file(
        file.filename
    ):

        return render_template(
            "index.html",
            error=(
                "Please upload a JPG, JPEG or PNG image."
            )
        )


    # --------------------------------------------------------
    # SECURE FILENAME
    # --------------------------------------------------------

    filename = secure_filename(
        file.filename
    )

    upload_path = (
        UPLOAD_FOLDER
        / filename
    )

    file.save(
        str(upload_path)
    )


    # --------------------------------------------------------
    # OPEN IMAGE
    # --------------------------------------------------------

    try:

        image = Image.open(
            upload_path
        ).convert("RGB")

    except Exception:

        return render_template(
            "index.html",
            error=(
                "The uploaded file is not a valid image."
            )
        )


    # --------------------------------------------------------
    # PREPROCESS
    # --------------------------------------------------------

    input_array = preprocess_image(
        image
    )


    # --------------------------------------------------------
    # PREDICTION + GRAD-CAM
    # --------------------------------------------------------

    try:
    # Normal model prediction
    prediction = model.predict(
        input_array,
        verbose=0
    )

    probability = float(prediction[0][0])

    # Generate Grad-CAM
    heatmap, _ = generate_gradcam(
        input_array
    )

except Exception as error:

    print(f"Prediction error: {error}")

    return render_template(
        "index.html",
        error=f"Analysis failed: {error}"
    )


    # --------------------------------------------------------
    # CLASSIFICATION
    # --------------------------------------------------------

    if probability >= THRESHOLD:

        predicted_class = "Tumor"

        confidence = (
            probability * 100
        )

    else:

        predicted_class = "No Tumor"

        confidence = (
            (1 - probability)
            * 100
        )


    tumor_probability = (
        probability * 100
    )

    no_tumor_probability = (
        (1 - probability)
        * 100
    )


    # --------------------------------------------------------
    # OUTPUT FILENAMES
    # --------------------------------------------------------

    stem = Path(
        filename
    ).stem

    overlay_filename = (
        f"{stem}_gradcam.jpg"
    )

    heatmap_filename = (
        f"{stem}_heatmap.jpg"
    )


    overlay_path = (
        OUTPUT_FOLDER
        / overlay_filename
    )

    heatmap_path = (
        OUTPUT_FOLDER
        / heatmap_filename
    )


    # --------------------------------------------------------
    # SAVE GRAD-CAM
    # --------------------------------------------------------

    overlay = create_gradcam_overlay(
        image,
        heatmap
    )

    overlay.save(
        overlay_path,
        quality=95
    )

    save_heatmap(
        heatmap,
        heatmap_path
    )


    # --------------------------------------------------------
    # RETURN RESULT
    # --------------------------------------------------------

    return render_template(
        "index.html",

        result=True,

        filename=filename,

        predicted_class=predicted_class,

        confidence=round(
            confidence,
            2
        ),

        tumor_probability=round(
            tumor_probability,
            2
        ),

        no_tumor_probability=round(
            no_tumor_probability,
            2
        ),

        original_image=url_for(
            "static",
            filename=f"uploads/{filename}"
        ),

        gradcam_image=url_for(
            "output_file",
            filename=overlay_filename
        ),

        heatmap_image=url_for(
            "output_file",
            filename=heatmap_filename
        )
    )


# ============================================================
# SERVE OUTPUT FILES
# ============================================================

@app.route(
    "/outputs/<filename>"
)
def output_file(filename):

    from flask import send_from_directory

    return send_from_directory(
        str(OUTPUT_FOLDER),
        filename
    )


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 70)
    print("FLASK SERVER STARTING")
    print("=" * 70)
    print()
    print(
        "Open: http://127.0.0.1:5000"
    )
    print()

    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )