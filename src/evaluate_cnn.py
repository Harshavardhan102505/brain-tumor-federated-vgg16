from pathlib import Path

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "preprocessed"
MODEL_PATH = PROJECT_ROOT / "models" / "custom_cnn.keras"
RESULTS_DIR = PROJECT_ROOT / "results"

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD DATA
# ============================================================

def load_test_data():

    images = []
    labels = []

    tumor_dir = DATA_DIR / "test" / "Tumor"
    no_tumor_dir = DATA_DIR / "test" / "No Tumor"

    for file in tumor_dir.glob("*.npy"):

        images.append(np.load(file))
        labels.append(1)

    for file in no_tumor_dir.glob("*.npy"):

        images.append(np.load(file))
        labels.append(0)

    images = np.array(
        images,
        dtype=np.float32
    )

    labels = np.array(
        labels,
        dtype=np.int32
    )

    return images, labels


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("CUSTOM CNN EVALUATION")
    print("=" * 60)

    # Load model
    print("\nLoading trained CNN...")

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    print("Model loaded successfully.")

    # Load test data
    print("\nLoading test dataset...")

    X_test, y_test = load_test_data()

    print(
        f"Test images: {len(X_test)}"
    )

    # Predictions
    print("\nGenerating predictions...")

    probabilities = model.predict(
        X_test,
        verbose=1
    )

    probabilities = probabilities.ravel()

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    # Metrics
    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)

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

    # Classification report
    print("\nClassification Report:")

    report = classification_report(
        y_test,
        predictions,
        target_names=[
            "No Tumor",
            "Tumor"
        ],
        zero_division=0
    )

    print(report)

    # Save classification report
    report_path = (
        RESULTS_DIR /
        "cnn_classification_report.txt"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(report)

        file.write(
            f"\nAccuracy : {accuracy:.4f}\n"
        )

        file.write(
            f"Precision: {precision:.4f}\n"
        )

        file.write(
            f"Recall   : {recall:.4f}\n"
        )

        file.write(
            f"F1 Score : {f1:.4f}\n"
        )

    # Confusion matrix
    cm = confusion_matrix(
        y_test,
        predictions
    )

    print("\nConfusion Matrix:")

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
        "Custom CNN Confusion Matrix"
    )

    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR /
        "cnn_confusion_matrix.png",
        dpi=300
    )

    plt.close()

    print(
        "\nSaved:"
        "\nresults/cnn_confusion_matrix.png"
        "\nresults/cnn_classification_report.txt"
    )

    print("\n" + "=" * 60)
    print("CNN EVALUATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()