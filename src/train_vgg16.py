from pathlib import Path

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "preprocessed"
MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10

SEED = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)


# ============================================================
# LOAD DATA
# ============================================================

def load_dataset(split):

    images = []
    labels = []

    tumor_dir = DATA_DIR / split / "Tumor"
    no_tumor_dir = DATA_DIR / split / "No Tumor"

    print(f"\nLoading {split} dataset...")

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
        dtype=np.float32
    )

    indices = np.random.permutation(
        len(images)
    )

    images = images[indices]
    labels = labels[indices]

    print(
        f"{split}: {len(images)} images"
    )

    return images, labels


# ============================================================
# CREATE VGG16 MODEL
# ============================================================

def create_model():

    print("\nLoading VGG16 ImageNet weights...")

    base_model = VGG16(
        weights="imagenet",
        include_top=False,
        input_shape=(
            IMG_SIZE,
            IMG_SIZE,
            3
        )
    )

    # Freeze VGG16 convolutional layers
    base_model.trainable = False

    model = models.Sequential([

        base_model,

        layers.GlobalAveragePooling2D(),

        layers.Dense(
            128,
            activation="relu"
        ),

        layers.Dropout(0.5),

        layers.Dense(
            1,
            activation="sigmoid"
        )
    ])

    model.compile(

        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.0001
        ),

        loss="binary_crossentropy",

        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(
                name="precision"
            ),
            tf.keras.metrics.Recall(
                name="recall"
            )
        ]
    )

    return model


# ============================================================
# SAVE TRAINING GRAPHS
# ============================================================

def save_training_graphs(history):

    # Accuracy
    plt.figure(figsize=(8, 6))

    plt.plot(
        history.history["accuracy"],
        label="Training Accuracy"
    )

    plt.plot(
        history.history["val_accuracy"],
        label="Validation Accuracy"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    plt.title(
        "VGG16 Training and Validation Accuracy"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "vgg16_accuracy.png",
        dpi=300
    )

    plt.close()

    # Loss
    plt.figure(figsize=(8, 6))

    plt.plot(
        history.history["loss"],
        label="Training Loss"
    )

    plt.plot(
        history.history["val_loss"],
        label="Validation Loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.title(
        "VGG16 Training and Validation Loss"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "vgg16_loss.png",
        dpi=300
    )

    plt.close()

    print("\nTraining graphs saved.")


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("VGG16 TRANSFER LEARNING")
    print("=" * 60)

    # Load datasets
    X_train, y_train = load_dataset("train")

    X_val, y_val = load_dataset("val")

    X_test, y_test = load_dataset("test")

    print("\nDataset shapes:")

    print("X_train:", X_train.shape)
    print("y_train:", y_train.shape)

    print("X_val:", X_val.shape)
    print("y_val:", y_val.shape)

    print("X_test:", X_test.shape)
    print("y_test:", y_test.shape)

    # Create model
    model = create_model()

    print("\nVGG16 model summary:")

    model.summary()

    # Train
    print("\nStarting VGG16 training...")

    history = model.fit(

        X_train,
        y_train,

        validation_data=(
            X_val,
            y_val
        ),

        epochs=EPOCHS,

        batch_size=BATCH_SIZE,

        verbose=1
    )

    # Save graphs
    save_training_graphs(history)

    # Evaluate
    print("\nEvaluating VGG16 on test data...")

    results = model.evaluate(
        X_test,
        y_test,
        verbose=1
    )

    print("\nTest Results:")

    for name, value in zip(
        model.metrics_names,
        results
    ):

        print(
            f"{name}: {value:.4f}"
        )

    # Save model
    model_path = (
        MODEL_DIR /
        "vgg16_brain_tumor.keras"
    )

    model.save(model_path)

    print("\nModel saved to:")

    print(model_path)

    print("\n" + "=" * 60)
    print("VGG16 TRAINING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()