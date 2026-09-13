import numpy as np
import streamlit as st
import tensorflow as tf
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "federated_vgg16_round5.keras"
)

IMG_SIZE = 224
THRESHOLD = 0.50


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Brain Tumor Detection",
    page_icon="🧠",
    layout="wide"
)


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    /* Main background */
    .stApp {
        background-color: #0b1220;
    }

    /* Main content */
    .block-container {
        max-width: 1150px;
        padding-top: 35px;
        padding-bottom: 40px;
    }

    /* All normal text */
    body,
    p,
    label,
    span,
    div {
        color: #f8fafc;
    }

    /* Title */
    h1 {
        color: #ffffff !important;
        text-align: center;
        font-size: 44px !important;
        font-weight: 800 !important;
    }

    /* Headings */
    h2,
    h3 {
        color: #ffffff !important;
    }

    /* Subtitle */
    .subtitle-text {
        text-align: center;
        color: #aebbd0;
        font-size: 18px;
        margin-bottom: 35px;
    }

    /* Upload area */
    [data-testid="stFileUploader"] {
        background-color: #162238;
        border: 2px dashed #3b82f6;
        border-radius: 15px;
        padding: 15px;
    }

    [data-testid="stFileUploaderDropzone"] {
        background-color: #162238 !important;
    }

    [data-testid="stFileUploaderDropzoneInstructions"] {
        color: #ffffff !important;
    }

    [data-testid="stFileUploaderDropzoneInstructions"] span {
        color: #ffffff !important;
    }

    [data-testid="stFileUploaderDropzoneInstructions"] small {
        color: #aebbd0 !important;
    }

    [data-testid="stFileUploader"] button {
        background-color: #2563eb !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background-color: #121c2e;
        border: 1px solid #2b3b55;
        border-radius: 14px;
        padding: 20px;
    }

    [data-testid="stMetricLabel"] {
        color: #aebbd0 !important;
    }

    [data-testid="stMetricValue"] {
        color: #ffffff !important;
    }

    /* Success alert */
    [data-testid="stAlert"] {
        border-radius: 12px;
    }

    /* Images */
    [data-testid="stImage"] {
        background-color: #121c2e;
        border-radius: 12px;
        padding: 8px;
    }

    /* Footer */
    .footer-text {
        text-align: center;
        color: #8291a8;
        font-size: 14px;
        padding-top: 25px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# TITLE
# ============================================================

st.title("🧠 Brain Tumor Detection")

st.markdown(
    '<p class="subtitle-text">'
    'Brain MRI Classification and Grad-CAM Visualization'
    '</p>',
    unsafe_allow_html=True
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Model file not found:\n{MODEL_PATH}"
        )

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    return model


try:

    model = load_model()

except Exception as e:

    st.error("Unable to load the trained model.")

    st.code(str(e))

    st.stop()


# ============================================================
# GET VGG16
# ============================================================

try:

    vgg16 = model.layers[0]

    last_conv_layer = vgg16.get_layer(
        "block5_conv3"
    )

except Exception as e:

    st.error("Unable to initialize Grad-CAM.")

    st.code(str(e))

    st.stop()


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
# PREPROCESS
# ============================================================

def preprocess_image(image):

    image = image.convert("RGB")

    image = image.resize(
        (IMG_SIZE, IMG_SIZE)
    )

    image_array = np.array(
        image,
        dtype=np.float32
    )

    image_array = image_array / 255.0

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    return image_array


# ============================================================
# GRAD-CAM
# ============================================================

def generate_gradcam(input_array):

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

        tumor_probability = predictions[:, 0]

    gradients = tape.gradient(
        tumor_probability,
        conv_outputs
    )

    if gradients is None:

        raise RuntimeError(
            "Grad-CAM gradients could not be calculated."
        )

    pooled_gradients = tf.reduce_mean(
        gradients,
        axis=(1, 2)
    )

    conv_outputs = conv_outputs[0]

    pooled_gradients = pooled_gradients[0]

    weighted_features = (
        conv_outputs
        * pooled_gradients
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

        heatmap = heatmap / maximum

    return (
        heatmap.numpy(),
        float(
            tumor_probability.numpy()[0]
        )
    )


# ============================================================
# CREATE GRAD-CAM OVERLAY
# ============================================================

def create_overlay(
    image,
    heatmap,
    alpha=0.45
):

    heatmap_uint8 = (
        heatmap * 255
    ).astype(np.uint8)

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

    cmap = plt.get_cmap("jet")

    colored_heatmap = cmap(
        heatmap_array / 255.0
    )

    colored_heatmap = (
        colored_heatmap[:, :, :3]
        * 255
    ).astype(np.uint8)

    original_array = np.array(
        image
    ).astype(np.float32)

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
    ).astype(np.uint8)

    return Image.fromarray(
        overlay
    )


# ============================================================
# UPLOAD SECTION
# ============================================================

st.subheader("📤 Upload Brain MRI")

st.write(
    "Select a JPG, JPEG or PNG brain MRI image."
)

uploaded_file = st.file_uploader(
    "Browse MRI Image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)


# ============================================================
# WAIT FOR IMAGE
# ============================================================

if uploaded_file is None:

    st.info(
        "Please upload an MRI image to begin the analysis."
    )

    st.markdown(
        '<div class="footer-text">'
        'Federated Learning Based Brain Tumor Detection'
        '<br>'
        'VGG16 • FedAvg • Grad-CAM'
        '</div>',
        unsafe_allow_html=True
    )

    st.stop()


# ============================================================
# PROCESS IMAGE
# ============================================================

try:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    input_array = preprocess_image(
        image
    )

    # --------------------------------------------------------
    # ANALYSIS
    # --------------------------------------------------------

    with st.spinner(
        "Analyzing MRI image..."
    ):

        heatmap, probability = (
            generate_gradcam(
                input_array
            )
        )

    # --------------------------------------------------------
    # CLASSIFICATION
    # --------------------------------------------------------

    if probability >= THRESHOLD:

        predicted_class = "Tumor"

        confidence = probability * 100

    else:

        predicted_class = "No Tumor"

        confidence = (
            1 - probability
        ) * 100


    # ========================================================
    # RESULT
    # ========================================================

    st.divider()

    st.subheader("🔍 Analysis Result")

    image_col, result_col = st.columns(
        2,
        gap="large"
    )


    # --------------------------------------------------------
    # MRI
    # --------------------------------------------------------

    with image_col:

        st.markdown("### 📷 Uploaded MRI")

        st.image(
            image,
            caption=uploaded_file.name
        )


    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    with result_col:

        st.markdown("### Prediction")

        if predicted_class == "Tumor":

            st.error(
                f"⚠️ Tumor Detected\n\n"
                f"Confidence: {confidence:.2f}%"
            )

        else:

            st.success(
                f"✅ No Tumor Detected\n\n"
                f"Confidence: {confidence:.2f}%"
            )

        st.write("")

        st.metric(
            "Predicted Class",
            predicted_class
        )

        st.metric(
            "Confidence",
            f"{confidence:.2f}%"
        )


    # ========================================================
    # PROBABILITY
    # ========================================================

    st.divider()

    st.subheader("📊 Prediction Probability")

    probability_col1, probability_col2 = (
        st.columns(2)
    )

    with probability_col1:

        st.metric(
            "Tumor Probability",
            f"{probability * 100:.2f}%"
        )

    with probability_col2:

        st.metric(
            "No Tumor Probability",
            f"{(1 - probability) * 100:.2f}%"
        )

    st.write("Tumor Probability")

    st.progress(
        float(
            np.clip(
                probability,
                0.0,
                1.0
            )
        )
    )


    # ========================================================
    # GRAD-CAM
    # ========================================================

    st.divider()

    st.subheader("🔥 Grad-CAM Visualization")

    st.write(
        "The highlighted regions show areas that contributed "
        "to the model prediction."
    )

    overlay = create_overlay(
        image,
        heatmap
    )


    cam_col1, cam_col2 = st.columns(
        2,
        gap="large"
    )


    with cam_col1:

        st.markdown("### Original MRI")

        st.image(
            image
        )


    with cam_col2:

        st.markdown("### 🔥 Grad-CAM Overlay")

        st.image(
            overlay
        )


    # ========================================================
    # HEATMAP
    # ========================================================

    st.markdown("### 🌡️ Attention Heatmap")

    st.image(
        heatmap,
        clamp=True
    )


    # ========================================================
    # DISCLAIMER
    # ========================================================

    


except Exception as e:

    st.error(
        "An error occurred while analyzing the MRI image."
    )

    st.code(
        str(e)
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
    '<div class="footer-text">'
    '<b>Federated Learning Based Brain Tumor Detection</b>'
    '<br>'
    'VGG16 • FedAvg • Grad-CAM'
    '</div>',
    unsafe_allow_html=True
)