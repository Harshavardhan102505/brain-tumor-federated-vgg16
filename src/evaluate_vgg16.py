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
MODEL_PATH = PROJECT_ROOT / "models" / "vgg16_brain_tumor.keras"
RESULTS_DIR = PROJECT_ROOT / "results"

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD TEST DATA
# ============================================================

def load_test_data():

    images = []
    labels = []

    tumor_dir = DATA_DIR / "test" / "Tumor"
    no_tumor_dir = DATA_DIR / "test" / "No Tumor"

    print("\nLoading test images...")

    # Tumor = 1
    for file in tumor_dir.glob("*.npy"):

        images.append(np.load(file))
        labels.append(1)

    # No Tumor = 0
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
    print("VGG16 MODEL EVALUATION")
    print("=" * 60)

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print("\nLoading trained VGG16 model...")

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    print("VGG16 model loaded successfully.")

    # --------------------------------------------------------
    # Load test data
    # --------------------------------------------------------

    X_test, y_test = load_test_data()

    print(
        f"Test images: {len(X_test)}"
    )

    print(
        f"Test shape: {X_test.shape}"
    )

    # --------------------------------------------------------
    # Generate predictions
    # --------------------------------------------------------

    print("\nGenerating predictions...")

    probabilities = model.predict(
        X_test,
        batch_size=32,
        verbose=1
    )

    probabilities = probabilities.ravel()

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    # --------------------------------------------------------
    # Calculate metrics
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("VGG16 TEST RESULTS")
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

    # --------------------------------------------------------
    # Classification Report
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)

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

    # Save report
    report_path = (
        RESULTS_DIR /
        "vgg16_classification_report.txt"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "VGG16 CLASSIFICATION REPORT\n"
        )

        file.write(
            "=" * 50 + "\n\n"
        )

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

    # --------------------------------------------------------
    # Confusion Matrix
    # --------------------------------------------------------

    cm = confusion_matrix(
        y_test,
        predictions
    )

    print("\n" + "=" * 60)
    print("CONFUSION MATRIX")
    print("=" * 60)

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
        "VGG16 Confusion Matrix"
    )

    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR /
        "vgg16_confusion_matrix.png",
        dpi=300
    )

    plt.close()

    # --------------------------------------------------------
    # Save numerical results
    # --------------------------------------------------------

    results_path = (
        RESULTS_DIR /
        "vgg16_results.txt"
    )

    with open(
        results_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "VGG16 TEST RESULTS\n"
        )

        file.write(
            "=" * 50 + "\n\n"
        )

        file.write(
            f"Accuracy : {accuracy:.4f}\n"
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

        file.write(
            "\nConfusion Matrix:\n"
        )

        file.write(
            str(cm)
        )

    print("\nSaved files:")

    print(
        "results/vgg16_classification_report.txt"
    )

    print(
        "results/vgg16_confusion_matrix.png"
    )

    print(
        "results/vgg16_results.txt"
    )

    print("\n" + "=" * 60)
    print("VGG16 EVALUATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()