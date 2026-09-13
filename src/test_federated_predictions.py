import tensorflow as tf
import numpy as np
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "federated_vgg16_round5.keras"
)

TEST_DIR = (
    PROJECT_ROOT
    / "data"
    / "preprocessed"
    / "test"
)

IMG_SIZE = 224


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 70)
print("FEDERATED VGG16 PREDICTION DIAGNOSTIC")
print("=" * 70)

print("\nLoading model...")

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

print("Model loaded successfully.")


# ============================================================
# LOAD IMAGE
# ============================================================

def load_image(path):

    image = np.load(path)

    image = tf.convert_to_tensor(
        image,
        dtype=tf.float32
    )

    image = tf.image.resize(
        image,
        [IMG_SIZE, IMG_SIZE]
    )

    image = tf.expand_dims(
        image,
        axis=0
    )

    return image


# ============================================================
# PREDICTION
# ============================================================

def predict_folder(folder_name, actual_label):

    folder = TEST_DIR / folder_name

    files = sorted(
        folder.glob("*.npy")
    )[:20]

    print("\n" + "-" * 70)
    print(f"ACTUAL CLASS: {folder_name}")
    print("-" * 70)

    correct = 0

    for file in files:

        image = load_image(file)

        probability = float(
            model.predict(
                image,
                verbose=0
            )[0][0]
        )

        predicted_label = (
            1 if probability >= 0.5
            else 0
        )

        predicted_name = (
            "Tumor"
            if predicted_label == 1
            else "No Tumor"
        )

        actual_name = (
            "Tumor"
            if actual_label == 1
            else "No Tumor"
        )

        if predicted_label == actual_label:
            correct += 1

        print(
            f"{file.name:25s} "
            f"Actual: {actual_name:10s} "
            f"Prediction: {predicted_name:10s} "
            f"Probability: {probability:.4f}"
        )

    print(
        f"\nCorrect: {correct}/{len(files)}"
    )


# ============================================================
# RUN TEST
# ============================================================

predict_folder(
    "No Tumor",
    0
)

predict_folder(
    "Tumor",
    1
)

print("\n" + "=" * 70)
print("DIAGNOSTIC COMPLETE")
print("=" * 70)