from pathlib import Path

import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TEST_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "test"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "federated_vgg16_round1.keras"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

IMG_SIZE = 224
BATCH_SIZE = 32


# ============================================================
# LOAD TEST DATA
# ============================================================

def get_test_files():

    tumor_dir = (
        TEST_DIR / "Tumor"
    )

    no_tumor_dir = (
        TEST_DIR / "No Tumor"
    )

    tumor_files = []
    no_tumor_files = []

    for extension in [
        "*.jpg",
        "*.jpeg",
        "*.png"
    ]:

        tumor_files.extend(
            tumor_dir.glob(extension)
        )

        no_tumor_files.extend(
            no_tumor_dir.glob(extension)
        )

    files = (
        tumor_files +
        no_tumor_files
    )

    labels = (
        [1] * len(tumor_files)
        +
        [0] * len(no_tumor_files)
    )

    return files, np.array(
        labels,
        dtype=np.int32
    )


# ============================================================
# LOAD IMAGE
# ============================================================

def load_image(path):

    image = tf.io.read_file(
        path
    )

    image = tf.image.decode_image(
        image,
        channels=3,
        expand_animations=False
    )

    image.set_shape(
        [None, None, 3]
    )

    image = tf.image.resize(
        image,
        [IMG_SIZE, IMG_SIZE]
    )

    image = tf.cast(
        image,
        tf.float32
    ) / 255.0

    return image


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 65)
    print("FEDERATED VGG16 EVALUATION")
    print("=" * 65)

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print(
        "\nLoading federated global model..."
    )

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    print(
        "Global model loaded successfully."
    )

    # --------------------------------------------------------
    # Load test files
    # --------------------------------------------------------

    files, y_test = get_test_files()

    print(
        f"\nTest images: {len(files)}"
    )

    print(
        f"Tumor: {sum(y_test == 1)}"
    )

    print(
        f"No Tumor: {sum(y_test == 0)}"
    )

    # --------------------------------------------------------
    # Generate predictions
    # --------------------------------------------------------

    print(
        "\nGenerating predictions..."
    )

    predictions = []

    for start in range(
        0,
        len(files),
        BATCH_SIZE
    ):

        batch_files = files[
            start:start + BATCH_SIZE
        ]

        batch_images = []

        for file in batch_files:

            image = load_image(
                str(file)
            )

            batch_images.append(
                image
            )

        batch_images = tf.stack(
            batch_images
        )

        batch_predictions = (
            model.predict(
                batch_images,
                verbose=0
            )
        )

        predictions.extend(
            batch_predictions.ravel()
        )

    probabilities = np.array(
        predictions
    )

    y_pred = (
        probabilities >= 0.5
    ).astype(int)

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("FEDERATED VGG16 ROUND 1 RESULTS")
    print("=" * 65)

    print(
        f"\nAccuracy : {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall   : {recall:.4f}"
    )

    print(
        f"F1 Score : {f1:.4f}"
    )

    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("CLASSIFICATION REPORT")
    print("=" * 65)

    report = classification_report(
        y_test,
        y_pred,
        target_names=[
            "No Tumor",
            "Tumor"
        ],
        zero_division=0
    )

    print(report)

    # Save report
    report_path = (
        RESULTS_DIR
        / "federated_round1_report.txt"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "FEDERATED VGG16 ROUND 1 RESULTS\n"
        )

        f.write(
            "=" * 50 + "\n\n"
        )

        f.write(
            f"Accuracy : {accuracy:.4f}\n"
        )

        f.write(
            f"Precision: {precision:.4f}\n"
        )

        f.write(
            f"Recall   : {recall:.4f}\n"
        )

        f.write(
            f"F1 Score : {f1:.4f}\n\n"
        )

        f.write(
            report
        )

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    print("\n" + "=" * 65)
    print("CONFUSION MATRIX")
    print("=" * 65)

    print(cm)

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[
            "No Tumor",
            "Tumor"
        ]
    )

    display.plot()

    plt.title(
        "Federated VGG16 - Round 1"
    )

    plt.tight_layout()

    cm_path = (
        RESULTS_DIR
        / "federated_round1_confusion_matrix.png"
    )

    plt.savefig(
        cm_path,
        dpi=300
    )

    plt.close()

    # --------------------------------------------------------
    # Save numerical results
    # --------------------------------------------------------

    results_path = (
        RESULTS_DIR
        / "federated_round1_results.txt"
    )

    with open(
        results_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "Federated VGG16 Round 1\n"
        )

        f.write(
            f"Accuracy : {accuracy:.4f}\n"
        )

        f.write(
            f"Precision: {precision:.4f}\n"
        )

        f.write(
            f"Recall   : {recall:.4f}\n"
        )

        f.write(
            f"F1 Score : {f1:.4f}\n\n"
        )

        f.write(
            "Confusion Matrix:\n"
        )

        f.write(
            str(cm)
        )

    print(
        "\nResults saved successfully."
    )

    print(
        "\n" + "=" * 65
    )

    print(
        "FEDERATED ROUND 1 EVALUATION COMPLETE"
    )

    print(
        "=" * 65
    )


if __name__ == "__main__":

    main()