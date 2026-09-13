from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16

# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CLIENT_DIR = PROJECT_ROOT / "data" / "federated_clients"
MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

NUM_CLIENTS = 4
START_ROUND = 2
END_ROUND = 5

LOCAL_EPOCHS = 1
BATCH_SIZE = 32
IMG_SIZE = 224
SEED = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)


# ============================================================
# MODEL
# ============================================================

def create_model():

    base_model = VGG16(
        weights="imagenet",
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )

    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.0001
        ),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


# ============================================================
# CLIENT FILES
# ============================================================

def get_client_files(client_number):

    client_path = (
        CLIENT_DIR / f"client_{client_number}"
    )

    tumor_dir = client_path / "Tumor"
    no_tumor_dir = client_path / "No Tumor"

    tumor_files = []
    no_tumor_files = []

    for extension in ["*.jpg", "*.jpeg", "*.png"]:

        tumor_files.extend(
            tumor_dir.glob(extension)
        )

        no_tumor_files.extend(
            no_tumor_dir.glob(extension)
        )

    files = tumor_files + no_tumor_files

    labels = (
        [1] * len(tumor_files)
        +
        [0] * len(no_tumor_files)
    )

    indices = np.random.permutation(len(files))

    files = [files[i] for i in indices]
    labels = [labels[i] for i in indices]

    return files, labels


# ============================================================
# IMAGE LOADING
# ============================================================

def load_image(path, label):

    image = tf.io.read_file(path)

    image = tf.image.decode_image(
        image,
        channels=3,
        expand_animations=False
    )

    image.set_shape([None, None, 3])

    image = tf.image.resize(
        image,
        [IMG_SIZE, IMG_SIZE]
    )

    image = tf.cast(
        image,
        tf.float32
    ) / 255.0

    label = tf.cast(
        label,
        tf.float32
    )

    return image, label


# ============================================================
# DATASET
# ============================================================

def create_dataset(files, labels):

    paths = [str(file) for file in files]

    dataset = tf.data.Dataset.from_tensor_slices(
        (paths, labels)
    )

    dataset = dataset.shuffle(
        len(paths),
        seed=SEED
    )

    dataset = dataset.map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.prefetch(
        tf.data.AUTOTUNE
    )

    return dataset


# ============================================================
# FEDAVG
# ============================================================

def fedavg(client_weights, client_sizes):

    total_samples = sum(client_sizes)

    averaged_weights = []

    for layer_weights in zip(*client_weights):

        weighted_layer = np.zeros_like(
            layer_weights[0]
        )

        for weights, size in zip(
            layer_weights,
            client_sizes
        ):

            weighted_layer += (
                weights *
                size /
                total_samples
            )

        averaged_weights.append(
            weighted_layer
        )

    return averaged_weights


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("MULTI-ROUND FEDERATED VGG16")
    print("=" * 70)

    # --------------------------------------------------------
    # Load Round 1 global model
    # --------------------------------------------------------

    round1_path = (
        MODEL_DIR /
        "federated_vgg16_round1.keras"
    )

    print(
        "\nLoading Round 1 global model..."
    )

    global_model = tf.keras.models.load_model(
        round1_path
    )

    global_weights = (
        global_model.get_weights()
    )

    print(
        "Round 1 model loaded successfully."
    )

    # --------------------------------------------------------
    # Training rounds
    # --------------------------------------------------------

    history_file = (
        RESULTS_DIR /
        "federated_round_history.txt"
    )

    with open(
        history_file,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "FEDERATED LEARNING ROUND HISTORY\n"
        )

        f.write(
            "=" * 50 + "\n\n"
        )

    for round_number in range(
        START_ROUND,
        END_ROUND + 1
    ):

        print("\n")
        print("=" * 70)
        print(
            f"FEDERATED ROUND {round_number}"
        )
        print("=" * 70)

        client_weights = []
        client_sizes = []

        # ----------------------------------------------------
        # Local client training
        # ----------------------------------------------------

        for client_number in range(
            1,
            NUM_CLIENTS + 1
        ):

            print("\n" + "-" * 60)

            print(
                f"CLIENT {client_number}"
            )

            print("-" * 60)

            files, labels = get_client_files(
                client_number
            )

            print(
                f"Training images: {len(files)}"
            )

            dataset = create_dataset(
                files,
                labels
            )

            local_model = create_model()

            # Start from current global model
            local_model.set_weights(
                global_weights
            )

            print(
                "Local training..."
            )

            local_model.fit(
                dataset,
                epochs=LOCAL_EPOCHS,
                verbose=1
            )

            print(
                f"Client {client_number} "
                "completed."
            )

            client_weights.append(
                local_model.get_weights()
            )

            client_sizes.append(
                len(files)
            )

            del local_model
            del dataset

        # ----------------------------------------------------
        # FedAvg
        # ----------------------------------------------------

        print("\n")
        print(
            "Performing FedAvg aggregation..."
        )

        global_weights = fedavg(
            client_weights,
            client_sizes
        )

        global_model.set_weights(
            global_weights
        )

        # ----------------------------------------------------
        # Save round model
        # ----------------------------------------------------

        round_model_path = (
            MODEL_DIR /
            f"federated_vgg16_round{round_number}.keras"
        )

        global_model.save(
            round_model_path
        )

        print(
            f"\nRound {round_number} model saved:"
        )

        print(
            round_model_path
        )

        # ----------------------------------------------------
        # Record history
        # ----------------------------------------------------

        with open(
            history_file,
            "a",
            encoding="utf-8"
        ) as f:

            f.write(
                f"Round {round_number}\n"
            )

            f.write(
                f"Client 1 samples: "
                f"{client_sizes[0]}\n"
            )

            f.write(
                f"Client 2 samples: "
                f"{client_sizes[1]}\n"
            )

            f.write(
                f"Client 3 samples: "
                f"{client_sizes[2]}\n"
            )

            f.write(
                f"Client 4 samples: "
                f"{client_sizes[3]}\n"
            )

            f.write("\n")

        print(
            f"\nFEDERATED ROUND "
            f"{round_number} COMPLETE"
        )

    print("\n")
    print("=" * 70)
    print("MULTI-ROUND FEDERATED TRAINING COMPLETE")
    print("=" * 70)

    print(
        "\nFinal model:"
    )

    print(
        MODEL_DIR /
        f"federated_vgg16_round{END_ROUND}.keras"
    )


if __name__ == "__main__":
    main()