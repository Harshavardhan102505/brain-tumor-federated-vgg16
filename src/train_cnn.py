from pathlib import Path
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers, models


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "preprocessed"
MODEL_DIR = PROJECT_ROOT / "models"

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 15

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

        image = np.load(file)

        images.append(image)
        labels.append(1)

    # No Tumor = 0
    for file in no_tumor_dir.glob("*.npy"):

        image = np.load(file)

        images.append(image)
        labels.append(0)

    images = np.array(images, dtype=np.float32)
    labels = np.array(labels, dtype=np.float32)

    # Shuffle
    indices = np.random.permutation(len(images))

    images = images[indices]
    labels = labels[indices]

    print(f"{split}: {len(images)} images")

    return images, labels


# ============================================================
# CNN MODEL
# ============================================================

def create_model():

    model = models.Sequential([

        layers.Input(
            shape=(IMG_SIZE, IMG_SIZE, 3)
        ),

        # Block 1
        layers.Conv2D(
            32,
            (3, 3),
            activation="relu",
            padding="same"
        ),

        layers.MaxPooling2D(
            (2, 2)
        ),

        # Block 2
        layers.Conv2D(
            64,
            (3, 3),
            activation="relu",
            padding="same"
        ),

        layers.MaxPooling2D(
            (2, 2)
        ),

        # Block 3
        layers.Conv2D(
            128,
            (3, 3),
            activation="relu",
            padding="same"
        ),

        layers.MaxPooling2D(
            (2, 2)
        ),

        # Classification
        layers.Flatten(),

        layers.Dense(
            128,
            activation="relu"
        ),

        layers.Dropout(
            0.5
        ),

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
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("CUSTOM CNN TRAINING")
    print("=" * 60)

    # Load data
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
    print("\nCreating CNN model...")

    model = create_model()

    model.summary()

    # Train
    print("\nStarting training...")

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

    # Test evaluation
    print("\nEvaluating on test dataset...")

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
        "custom_cnn.keras"
    )

    model.save(model_path)

    print("\nModel saved to:")

    print(model_path)

    print("\n" + "=" * 60)
    print("CNN TRAINING COMPLETE")
    print("=" * 60)

def save_training_graphs(history):

    import matplotlib.pyplot as plt

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
    plt.title("Custom CNN Accuracy")

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "cnn_accuracy.png",
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
    plt.title("Custom CNN Loss")

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "cnn_loss.png",
        dpi=300
    )

    plt.close()

    print("\nTraining graphs saved:")
    print("results/cnn_accuracy.png")
    print("results/cnn_loss.png")

if __name__ == "__main__":
    main()