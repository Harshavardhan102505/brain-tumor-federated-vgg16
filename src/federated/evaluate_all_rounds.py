from pathlib import Path

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = PROJECT_ROOT / "models"

TEST_DIR = PROJECT_ROOT / "data" / "processed" / "test"

RESULTS_DIR = PROJECT_ROOT / "results"

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

IMG_SIZE = 224
BATCH_SIZE = 32

START_ROUND = 1
END_ROUND = 5


# ============================================================
# GET TEST FILES
# ============================================================

def get_test_files():

    tumor_dir = TEST_DIR / "Tumor"
    no_tumor_dir = TEST_DIR / "No Tumor"

    tumor_files = []
    no_tumor_files = []

    for ext in ["*.jpg", "*.jpeg", "*.png"]:

        tumor_files.extend(
            tumor_dir.glob(ext)
        )

        no_tumor_files.extend(
            no_tumor_dir.glob(ext)
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
        str(path)
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
# PREDICT
# ============================================================

def predict_model(model, files):

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

            image = load_image(file)

            batch_images.append(
                image
            )

        batch_images = tf.stack(
            batch_images
        )

        batch_predictions = model.predict(
            batch_images,
            verbose=0
        )

        predictions.extend(
            batch_predictions.ravel()
        )

    probabilities = np.array(
        predictions
    )

    return (
        probabilities >= 0.5
    ).astype(int)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("FEDERATED ROUND-BY-ROUND EVALUATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Load test data
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

    if len(files) == 0:

        raise ValueError(
            "No test images found. "
            "Check TEST_DIR."
        )

    # --------------------------------------------------------
    # Store results
    # --------------------------------------------------------

    rounds = []

    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []

    # --------------------------------------------------------
    # Evaluate each round
    # --------------------------------------------------------

    for round_number in range(
        START_ROUND,
        END_ROUND + 1
    ):

        print("\n" + "=" * 70)

        print(
            f"EVALUATING ROUND {round_number}"
        )

        print("=" * 70)

        model_path = (
            MODEL_DIR /
            f"federated_vgg16_round{round_number}.keras"
        )

        if not model_path.exists():

            print(
                f"Model not found: {model_path}"
            )

            continue

        print(
            "Loading model..."
        )

        model = tf.keras.models.load_model(
            model_path
        )

        print(
            "Generating predictions..."
        )

        y_pred = predict_model(
            model,
            files
        )

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

        rounds.append(
            round_number
        )

        accuracies.append(
            accuracy
        )

        precisions.append(
            precision
        )

        recalls.append(
            recall
        )

        f1_scores.append(
            f1
        )

        print(
            f"\nRound {round_number} Results:"
        )

        print(
            f"Accuracy : {accuracy:.4f}"
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

        # ----------------------------------------------------
        # Final round confusion matrix
        # ----------------------------------------------------

        if round_number == END_ROUND:

            cm = confusion_matrix(
                y_test,
                y_pred
            )

            print(
                "\nFinal Confusion Matrix:"
            )

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
                "Federated VGG16 - Final Round"
            )

            plt.tight_layout()

            plt.savefig(
                RESULTS_DIR /
                "federated_final_confusion_matrix.png",
                dpi=300
            )

            plt.close()

        del model

    # --------------------------------------------------------
    # Save CSV-style results
    # --------------------------------------------------------

    results_file = (
        RESULTS_DIR /
        "federated_round_metrics.csv"
    )

    with open(
        results_file,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "Round,Accuracy,Precision,Recall,F1\n"
        )

        for i in range(
            len(rounds)
        ):

            f.write(
                f"{rounds[i]},"
                f"{accuracies[i]:.6f},"
                f"{precisions[i]:.6f},"
                f"{recalls[i]:.6f},"
                f"{f1_scores[i]:.6f}\n"
            )

    # --------------------------------------------------------
    # Print final table
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("FEDERATED LEARNING RESULTS")
    print("=" * 70)

    print(
        "\nRound | Accuracy | Precision | Recall | F1"
    )

    print(
        "-" * 55
    )

    for i in range(
        len(rounds)
    ):

        print(
            f"{rounds[i]:5d} | "
            f"{accuracies[i]:.4f}   | "
            f"{precisions[i]:.4f}    | "
            f"{recalls[i]:.4f} | "
            f"{f1_scores[i]:.4f}"
        )

    # --------------------------------------------------------
    # Accuracy graph
    # --------------------------------------------------------

    plt.figure()

    plt.plot(
        rounds,
        accuracies,
        marker="o"
    )

    plt.xlabel(
        "Federated Round"
    )

    plt.ylabel(
        "Accuracy"
    )

    plt.title(
        "Federated VGG16 Accuracy vs Round"
    )

    plt.xticks(rounds)

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR /
        "accuracy_vs_round.png",
        dpi=300
    )

    plt.close()

    # --------------------------------------------------------
    # Precision graph
    # --------------------------------------------------------

    plt.figure()

    plt.plot(
        rounds,
        precisions,
        marker="o"
    )

    plt.xlabel(
        "Federated Round"
    )

    plt.ylabel(
        "Precision"
    )

    plt.title(
        "Federated VGG16 Precision vs Round"
    )

    plt.xticks(rounds)

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR /
        "precision_vs_round.png",
        dpi=300
    )

    plt.close()

    # --------------------------------------------------------
    # Recall graph
    # --------------------------------------------------------

    plt.figure()

    plt.plot(
        rounds,
        recalls,
        marker="o"
    )

    plt.xlabel(
        "Federated Round"
    )

    plt.ylabel(
        "Recall"
    )

    plt.title(
        "Federated VGG16 Recall vs Round"
    )

    plt.xticks(rounds)

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR /
        "recall_vs_round.png",
        dpi=300
    )

    plt.close()

    # --------------------------------------------------------
    # F1 graph
    # --------------------------------------------------------

    plt.figure()

    plt.plot(
        rounds,
        f1_scores,
        marker="o"
    )

    plt.xlabel(
        "Federated Round"
    )

    plt.ylabel(
        "F1 Score"
    )

    plt.title(
        "Federated VGG16 F1 Score vs Round"
    )

    plt.xticks(rounds)

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR /
        "f1_vs_round.png",
        dpi=300
    )

    plt.close()

    # --------------------------------------------------------
    # Final message
    # --------------------------------------------------------

    print("\n" + "=" * 70)

    print(
        "ROUND-BY-ROUND EVALUATION COMPLETE"
    )

    print("=" * 70)

    print(
        "\nResults saved in:"
    )

    print(
        RESULTS_DIR
    )


if __name__ == "__main__":

    main()