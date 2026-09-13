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
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# CUSTOM UI
# ============================================================

st.markdown(
    """
    <style>

    /* =====================================================
       GLOBAL PAGE
       ===================================================== */

    .stApp {
        background: #0b1220;
        color: #f8fafc;
    }

    .main {
        background: #0b1220;
    }

    .block-container {
        max-width: 1200px;
        padding-top: 35px;
        padding-bottom: 40px;
    }


    /* =====================================================
       REMOVE SIDEBAR
       ===================================================== */

    [data-testid="stSidebar"] {
        display: none;
    }


    /* =====================================================
       MAIN TITLE
       ===================================================== */

    .title-container {
        text-align: center;
        padding: 10px 0 30px 0;
    }

    .main-title {
        color: #ffffff;
        font-size: 46px;
        font-weight: 800;
        letter-spacing: -1px;
        margin-bottom: 8px;
    }

    .main-subtitle {
        color: #aebbd0;
        font-size: 18px;
        font-weight: 400;
    }


    /* =====================================================
       UPLOAD CARD
       ===================================================== */

    .upload-card {
        background: #121c2e;
        border: 1px solid #263650;
        border-radius: 18px;
        padding: 30px;
        margin: 10px 0 28px 0;
        box-shadow: 0 8px 30px rgba(0,0,0,0.25);
    }

    .upload-heading {
        color: #ffffff;
        font-size: 25px;
        font-weight: 700;
        text-align: center;
        margin-bottom: 6px;
    }

    .upload-text {
        color: #aebbd0;
        text-align: center;
        font-size: 15px;
        margin-bottom: 20px;
    }


    /* =====================================================
       FILE UPLOADER
       ===================================================== */

    [data-testid="stFileUploader"] {
        background: #18243a;
        border: 2px dashed #3b82f6;
        border-radius: 14px;
        padding: 10px;
    }

    [data-testid="stFileUploader"] section {
        background: transparent !important;
    }

    [data-testid="stFileUploaderDropzone"] {
        background: #18243a !important;
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
        background: #2563eb !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
    }


    /* =====================================================
       SECTION HEADINGS
       ===================================================== */

    .section-title {
        color: #ffffff;
        font-size: 27px;
        font-weight: 750;
        margin-top: 28px;
        margin-bottom: 18px;
    }

    .section-subtitle {
        color: #aebbd0;
        font-size: 15px;
        margin-top: -10px;
        margin-bottom: 20px;
    }


    /* =====================================================
       IMAGE CARDS
       ===================================================== */

    .image-card {
        background: #121c2e;
        border: 1px solid #263650;
        border-radius: 16px;
        padding: 18px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.20);
    }

    .image-title {
        color: #ffffff;
        font-size: 18px;
        font-weight: 700;
        text-align: center;
        margin-bottom: 12px;
    }


    /* =====================================================
       RESULT CARD
       ===================================================== */

    .result-card {
        min-height: 210px;
        border-radius: 18px;
        padding: 30px 20px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        text-align: center;
    }

    .tumor-result {
        background: #35151a;
        border: 2px solid #ef4444;
    }

    .normal-result {
        background: #102c20;
        border: 2px solid #22c55e;
    }

    .result-icon {
        font-size: 48px;
        margin-bottom: 8px;
    }

    .result-title {
        font-size: 29px;
        font-weight: 800;
        margin-bottom: 12px;
    }

    .tumor-title {
        color: #ff6b6b;
    }

    .normal-title {
        color: #4ade80;
    }

    .confidence-text {
        color: #ffffff;
        font-size: 19px;
        font-weight: 600;
    }


    /* =====================================================
       METRIC CARDS
       ===================================================== */

    .metric-card {
        background: #121c2e;
        border: 1px solid #263650;
        border-radius: 15px;
        padding: 22px;
        text-align: center;
        margin-top: 10px;
    }

    .metric-value {
        color: #ffffff;
        font-size: 30px;
        font-weight: 800;
    }

    .metric-label {
        color: #aebbd0;
        font-size: 14px;
        margin-top: 5px;
    }


    /* =====================================================
       STREAMLIT METRICS
       ===================================================== */

    [data-testid="stMetric"] {
        background: #121c2e;
        border: 1px solid #263650;
        border-radius: 14px;
        padding: 18px;
    }

    [data-testid="stMetricLabel"] {
        color: #aebbd0 !important;
    }

    [data-testid="stMetricValue"] {
        color: #ffffff !important;
    }


    /* =====================================================
       PROGRESS BAR
       ===================================================== */

    [data-testid="stProgressBar"] {
        background-color: #263650 !important;
    }


    /* =====================================================
       INFO MESSAGE
       ===================================================== */

    [data-testid="stAlert"] {
        background: #152238;
        color: #ffffff;
        border-radius: 12px;
        border: 1px solid #334765;
    }


    /* =====================================================
       WARNING
       ===================================================== */

    .disclaimer {
        background: #211d12;
        border: 1px solid #8a6d20;
        color: #f5d77a;
        border-radius: 12px;
        padding: 15px 18px;
        margin-top: 30px;
        text-align: center;
        font-size: 14px;
    }


    /* =====================================================
       FOOTER
       ===================================================== */

    .footer {
        border-top: 1px solid #263650;
        margin-top: 45px;
        padding-top: 22px;
        text-align: center;
        color: #8291a8;
        font-size: 13px;
    }

    .footer-title {
        color: #dbe5f2;
        font-weight: 700;
        font-size: 15px;
        margin-bottom: 5px;
    }


    /* =====================================================
       SPINNER
       ===================================================== */

    .stSpinner > div {
        border-top-color: #3b82f6 !important;
    }


    /* =====================================================
       BUTTONS
       ===================================================== */

    .stButton > button {
        background: #2563eb;
        color: #ffffff;
        border: none;
        border-radius: 9px;
        font-weight: 600;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="title-container">

        <div class="main-title">
            🧠 Brain Tumor Detection
        </div>

        <div class="main-subtitle">
            Brain MRI Classification and Grad-CAM Visualization
        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Model not found:\n{MODEL_PATH}"
        )

    return tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )


try:

    model = load_model()

except Exception as error:

    st.error(
        "Unable to load the trained model."
    )

    st.code(
        str(error)
    )

    st.stop()


# ============================================================
# VGG16 SUBMODEL
# ============================================================

try:

    vgg16 = model.layers[0]

    last_conv_layer = vgg16.get_layer(
        "block5_conv3"
    )

except Exception as error:

    st.error(
        "Unable to initialize Grad-CAM."
    )

    st.code(
        str(error)
    )

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

        tumor_probability = (
            predictions[:, 0]
        )

    gradients = tape.gradient(
        tumor_probability,
        conv_outputs
    )

    if gradients is None:

        raise RuntimeError(
            "Could not calculate Grad-CAM gradients."
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
# CREATE OVERLAY
# ============================================================

def create_overlay(
    image,
    heatmap,
    alpha=0.45
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
# UPLOAD CARD
# ============================================================

st.markdown(
    """
    <div class="upload-card">

        <div class="upload-heading">
            📤 Upload Brain MRI
        </div>

        <div class="upload-text">
            Select a JPG, JPEG or PNG brain MRI image
            for analysis.
        </div>

    </div>
    """,
    unsafe_allow_html=True
)


uploaded_file = st.file_uploader(
    "Browse MRI Image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ],
    label_visibility="visible"
)


# ============================================================
# NO IMAGE
# ============================================================

if uploaded_file is None:

    st.info(
        "Please upload a brain MRI image to begin."
    )

    st.markdown(
        """
        <div class="footer">

            <div class="footer-title">
                Federated Learning Based Brain Tumor Detection
            </div>

            VGG16 • FedAvg • Grad-CAM

        </div>
        """,
        unsafe_allow_html=True
    )

    st.stop()


# ============================================================
# PROCESS IMAGE
# ============================================================

try:

    image = Image.open(
        uploaded_file
    ).convert(
        "RGB"
    )

    input_array = preprocess_image(
        image
    )

    # ========================================================
    # PREDICTION
    # ========================================================

    with st.spinner(
        "Analyzing MRI image..."
    ):

        heatmap, probability = (
            generate_gradcam(
                input_array
            )
        )

    # ========================================================
    # CLASSIFICATION
    # ========================================================

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

    # ========================================================
    # RESULT
    # ========================================================

    st.markdown(
        '<div class="section-title">'
        '🔍 Analysis Result'
        '</div>',
        unsafe_allow_html=True
    )

    result_col1, result_col2 = (
        st.columns(
            [1, 1],
            gap="large"
        )
    )

    # ========================================================
    # ORIGINAL MRI
    # ========================================================

    with result_col1:

        st.markdown(
            '<div class="image-title">'
            '📷 Uploaded MRI'
            '</div>',
            unsafe_allow_html=True
        )

        st.image(
            image,
            use_column_width=True
        )

    # ========================================================
    # RESULT
    # ========================================================

    with result_col2:

        if predicted_class == "Tumor":

            st.markdown(
                f"""
                <div class="result-card tumor-result">

                    <div class="result-icon">
                        ⚠️
                    </div>

                    <div class="result-title tumor-title">
                        Tumor Detected
                    </div>

                    <div class="confidence-text">
                        Confidence: {confidence:.2f}%
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                f"""
                <div class="result-card normal-result">

                    <div class="result-icon">
                        ✅
                    </div>

                    <div class="result-title normal-title">
                        No Tumor Detected
                    </div>

                    <div class="confidence-text">
                        Confidence: {confidence:.2f}%
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )


    # ========================================================
    # PROBABILITY
    # ========================================================

    st.markdown(
        '<div class="section-title">'
        '📊 Prediction Probability'
        '</div>',
        unsafe_allow_html=True
    )

    prob_col1, prob_col2 = (
        st.columns(2)
    )

    with prob_col1:

        st.markdown(
            f"""
            <div class="metric-card">

                <div class="metric-value">
                    {probability * 100:.2f}%
                </div>

                <div class="metric-label">
                    Tumor Probability
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    with prob_col2:

        st.markdown(
            f"""
            <div class="metric-card">

                <div class="metric-value">
                    {(1 - probability) * 100:.2f}%
                </div>

                <div class="metric-label">
                    No Tumor Probability
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    # ========================================================
    # PROBABILITY BAR
    # ========================================================

    st.write("")

    st.markdown(
        '<div class="section-subtitle">'
        'Tumor probability'
        '</div>',
        unsafe_allow_html=True
    )

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

    st.markdown(
        '<div class="section-title">'
        '🔥 Grad-CAM Visualization'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Highlighted regions indicate areas that contributed '
        'to the model prediction.'
        '</div>',
        unsafe_allow_html=True
    )


    # ========================================================
    # CREATE OVERLAY
    # ========================================================

    overlay = create_overlay(
        image,
        heatmap,
        alpha=0.45
    )


    # ========================================================
    # ORIGINAL / GRAD-CAM
    # ========================================================

    cam_col1, cam_col2 = (
        st.columns(
            2,
            gap="large"
        )
    )

    with cam_col1:

        st.markdown(
            '<div class="image-title">'
            'Original MRI'
            '</div>',
            unsafe_allow_html=True
        )

        st.image(
            image,
            use_column_width=True
        )

    with cam_col2:

        st.markdown(
            '<div class="image-title">'
            '🔥 Grad-CAM Overlay'
            '</div>',
            unsafe_allow_html=True
        )

        st.image(
            overlay,
            use_column_width=True
        )


    # ========================================================
    # HEATMAP
    # ========================================================

    st.markdown(
        '<div class="section-title">'
        '🌡️ Attention Heatmap'
        '</div>',
        unsafe_allow_html=True
    )

    st.image(
        heatmap,
        use_column_width=True,
        clamp=True
    )


    # ========================================================
    # DISCLAIMER
    # ========================================================

    st.markdown(
        """
        <div class="disclaimer">

            ⚕️ <b>Research Prototype:</b>
            This application is intended for academic and
            research demonstration purposes only and should
            not be used as a medical diagnostic tool.

        </div>
        """,
        unsafe_allow_html=True
    )


except Exception as error:

    st.error(
        "An error occurred while analyzing the MRI image."
    )

    st.code(
        str(error)
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">

        <div class="footer-title">
            Federated Learning Based Brain Tumor Detection
        </div>

        VGG16 • FedAvg • Grad-CAM

    </div>
    """,
    unsafe_allow_html=True
)