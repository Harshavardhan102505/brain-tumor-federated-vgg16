import gc
from pathlib import Path

import numpy as np
import tensorflow as tf

from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CLIENT_DIR = (
    PROJECT_ROOT
    / "data"
    / "federated_clients"
)

TEST_DIR = (
    PROJECT_ROOT
    / "data"
    / "preprocessed"
    / "test"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# Federated settings
NUM_CLIENTS = 4
NUM_ROUNDS = 5

# Local training
LOCAL_EPOCHS = 5
BATCH_SIZE = 32

# Image
IMG_SIZE = 224

# Fine-tuning learning rate
LEARNING_RATE = 1e-5

# Random seed
SEED = 42

# Balance each client
BALANCE_CLIENTS = True


np.random.seed(SEED)
tf.random.set_seed(SEED)


# ============================================================
# GPU / CPU CONFIGURATION
# ============================================================

def configure_device():

    gpus = tf.config.list_physical_devices(
        "GPU"
    )

    if gpus:

        print(
            "\nGPU detected."
        )

        print(
            "GPU devices:",
            gpus
        )

        for gpu in gpus:

            try:

                tf.config.experimental.set_memory_growth(
                    gpu,
                    True
                )

            except Exception:

                pass

    else:

        print(
            "\nNo GPU detected."
        )

        print(
            "Training will use CPU."
        )


# ============================================================
# CREATE VGG16 MODEL
# ============================================================

def create_model():

    print(
        "\nCreating VGG16 model..."
    )

    base_model = VGG16(

        weights="imagenet",

        include_top=False,

        input_shape=(
            IMG_SIZE,
            IMG_SIZE,
            3
        )
    )

    # --------------------------------------------------------
    # PARTIAL FINE-TUNING
    # --------------------------------------------------------

    base_model.trainable = True

    # Freeze early layers.
    #
    # Only the final four VGG16 layers
    # will be trainable.
    #
    for layer in base_model.layers[:-4]:

        layer.trainable = False


    # Keep BatchNormalization frozen.
    #
    # This prevents unstable batch-statistics
    # during small federated client training.
    #
    for layer in base_model.layers:

        if isinstance(
            layer,
            layers.BatchNormalization
        ):

            layer.trainable = False


    # --------------------------------------------------------
    # CLASSIFICATION HEAD
    # --------------------------------------------------------

    model = models.Sequential(

        [

            base_model,

            layers.GlobalAveragePooling2D(),

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

        ],

        name="Federated_VGG16"
    )


    # --------------------------------------------------------
    # COMPILE
    # --------------------------------------------------------

    model.compile(

        optimizer=tf.keras.optimizers.Adam(

            learning_rate=LEARNING_RATE

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


    # --------------------------------------------------------
    # PARAMETER INFORMATION
    # --------------------------------------------------------

    trainable_parameters = sum(

        int(
            tf.keras.backend.count_params(
                weight
            )
        )

        for weight in model.trainable_weights

    )


    total_parameters = sum(

        int(
            tf.keras.backend.count_params(
                weight
            )
        )

        for weight in model.weights

    )


    print(
        f"Trainable parameters: "
        f"{trainable_parameters:,}"
    )

    print(
        f"Total parameters    : "
        f"{total_parameters:,}"
    )


    return model


# ============================================================
# GET NPY FILES
# ============================================================

def get_npy_files(
    directory
):

    directory = Path(
        directory
    )

    files = [

        file

        for file in directory.glob(
            "*.npy"
        )

        if file.is_file()

    ]

    return sorted(
        files
    )


# ============================================================
# LOAD NPY IMAGE
# ============================================================

def load_npy_image(
    path
):

    image = np.load(
        path
    ).astype(
        np.float32
    )


    # --------------------------------------------------------
    # Grayscale image
    # --------------------------------------------------------

    if image.ndim == 2:

        image = np.stack(
            [
                image,
                image,
                image
            ],
            axis=-1
        )


    # --------------------------------------------------------
    # Validate dimensions
    # --------------------------------------------------------

    if image.ndim != 3:

        raise ValueError(

            f"Invalid image shape in "
            f"{path}: {image.shape}"

        )


    # --------------------------------------------------------
    # One channel -> three channels
    # --------------------------------------------------------

    if image.shape[-1] == 1:

        image = np.repeat(
            image,
            3,
            axis=-1
        )


    # --------------------------------------------------------
    # Validate channels
    # --------------------------------------------------------

    if image.shape[-1] != 3:

        raise ValueError(

            f"Expected 3 channels in "
            f"{path}, got "
            f"{image.shape}"

        )


    # --------------------------------------------------------
    # Resize if required
    # --------------------------------------------------------

    if image.shape[:2] != (
        IMG_SIZE,
        IMG_SIZE
    ):

        image = tf.image.resize(

            image,

            [
                IMG_SIZE,
                IMG_SIZE
            ]

        ).numpy()


    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    max_value = float(
        np.max(image)
    )

    if max_value > 1.0:

        image = (
            image / 255.0
        )


    image = np.clip(
        image,
        0.0,
        1.0
    )


    return image.astype(
        np.float32
    )


# ============================================================
# GET CLIENT FILES
# ============================================================

def get_client_files(
    client_number
):

    client_path = (

        CLIENT_DIR
        / f"client_{client_number}"

    )


    tumor_dir = (
        client_path
        / "Tumor"
    )

    no_tumor_dir = (
        client_path
        / "No Tumor"
    )


    tumor_files = get_npy_files(
        tumor_dir
    )

    no_tumor_files = get_npy_files(
        no_tumor_dir
    )


    if len(tumor_files) == 0:

        raise ValueError(

            f"Client {client_number} "
            f"has no Tumor .npy images."

        )


    if len(no_tumor_files) == 0:

        raise ValueError(

            f"Client {client_number} "
            f"has no No Tumor .npy images."

        )


    print(
        f"\nClient {client_number} "
        f"original distribution:"
    )

    print(
        f"  Tumor    : "
        f"{len(tumor_files)}"
    )

    print(
        f"  No Tumor : "
        f"{len(no_tumor_files)}"
    )

    print(
        f"  Total    : "
        f"{len(tumor_files) + len(no_tumor_files)}"
    )


    return (
        tumor_files,
        no_tumor_files
    )


# ============================================================
# BALANCE CLIENT DATA
# ============================================================

def create_balanced_client_data(
    client_number
):

    tumor_files, no_tumor_files = (
        get_client_files(
            client_number
        )
    )


    # --------------------------------------------------------
    # No balancing requested
    # --------------------------------------------------------

    if not BALANCE_CLIENTS:

        files = (
            tumor_files
            + no_tumor_files
        )

        labels = (

            [1] * len(tumor_files)

            +

            [0] * len(no_tumor_files)

        )

        return (
            files,
            labels
        )


    # --------------------------------------------------------
    # Undersample larger class
    # --------------------------------------------------------

    target_count = min(

        len(tumor_files),

        len(no_tumor_files)

    )


    rng = np.random.default_rng(

        SEED
        + client_number

    )


    tumor_indices = rng.choice(

        len(tumor_files),

        size=target_count,

        replace=False

    )


    no_tumor_indices = rng.choice(

        len(no_tumor_files),

        size=target_count,

        replace=False

    )


    balanced_tumor = [

        tumor_files[
            int(index)
        ]

        for index in tumor_indices

    ]


    balanced_no_tumor = [

        no_tumor_files[
            int(index)
        ]

        for index in no_tumor_indices

    ]


    files = (

        balanced_tumor
        +
        balanced_no_tumor

    )


    labels = (

        [1] * len(balanced_tumor)

        +

        [0] * len(balanced_no_tumor)

    )


    # --------------------------------------------------------
    # Shuffle balanced dataset
    # --------------------------------------------------------

    indices = rng.permutation(
        len(files)
    )


    files = [

        files[int(index)]

        for index in indices

    ]


    labels = [

        labels[int(index)]

        for index in indices

    ]


    print(
        f"\nClient {client_number} "
        f"BALANCED distribution:"
    )

    print(
        f"  Tumor    : "
        f"{len(balanced_tumor)}"
    )

    print(
        f"  No Tumor : "
        f"{len(balanced_no_tumor)}"
    )

    print(
        f"  Total    : "
        f"{len(files)}"
    )


    return (
        files,
        labels
    )


# ============================================================
# CREATE CLIENT DATASET
# ============================================================

def create_client_dataset(
    files,
    labels
):

    print(
        "\nLoading local images..."
    )


    images = []


    for path in files:

        images.append(
            load_npy_image(
                path
            )
        )


    images = np.asarray(

        images,

        dtype=np.float32

    )


    labels = np.asarray(

        labels,

        dtype=np.float32

    )


    dataset = (

        tf.data.Dataset
        .from_tensor_slices(
            (
                images,
                labels
            )
        )

    )


    dataset = dataset.shuffle(

        buffer_size=len(images),

        seed=SEED,

        reshuffle_each_iteration=True

    )


    dataset = dataset.batch(

        BATCH_SIZE

    )


    dataset = dataset.prefetch(

        tf.data.AUTOTUNE

    )


    return dataset


# ============================================================
# LOAD TEST DATASET
# ============================================================

def load_test_dataset():

    tumor_dir = (
        TEST_DIR
        / "Tumor"
    )

    no_tumor_dir = (
        TEST_DIR
        / "No Tumor"
    )


    tumor_files = get_npy_files(
        tumor_dir
    )

    no_tumor_files = get_npy_files(
        no_tumor_dir
    )


    if len(tumor_files) == 0:

        raise ValueError(
            "No Tumor test images found."
        )


    if len(no_tumor_files) == 0:

        raise ValueError(
            "No No Tumor test images found."
        )


    files = (

        tumor_files
        +
        no_tumor_files

    )


    labels = (

        [1] * len(tumor_files)

        +

        [0] * len(no_tumor_files)

    )


    print(
        "\n"
        + "=" * 70
    )

    print(
        "TEST DATASET"
    )

    print(
        "=" * 70
    )


    print(
        f"Tumor    : "
        f"{len(tumor_files)}"
    )

    print(
        f"No Tumor : "
        f"{len(no_tumor_files)}"
    )

    print(
        f"Total    : "
        f"{len(files)}"
    )


    print(
        "\nLoading test images..."
    )


    images = np.asarray(

        [
            load_npy_image(
                path
            )

            for path in files
        ],

        dtype=np.float32

    )


    labels = np.asarray(

        labels,

        dtype=np.int32

    )


    return (
        images,
        labels
    )


# ============================================================
# FEDAVG
# ============================================================

def fedavg(
    client_weights,
    client_sizes
):

    if len(client_weights) == 0:

        raise ValueError(
            "No client weights supplied."
        )


    if len(client_weights) != len(
        client_sizes
    ):

        raise ValueError(
            "Client weights and client "
            "sizes do not match."
        )


    total_samples = sum(
        client_sizes
    )


    if total_samples <= 0:

        raise ValueError(
            "Total samples must be greater "
            "than zero."
        )


    averaged_weights = []


    for layer_weights in zip(
        *client_weights
    ):

        weighted_average = (
            np.zeros_like(
                layer_weights[0],
                dtype=np.float32
            )
        )


        for weights, size in zip(

            layer_weights,

            client_sizes

        ):

            weighted_average += (

                weights.astype(
                    np.float32
                )

                *

                (
                    float(size)
                    /
                    float(total_samples)
                )

            )


        averaged_weights.append(
            weighted_average
        )


    return averaged_weights


# ============================================================
# EVALUATE GLOBAL MODEL
# ============================================================

def evaluate_global_model(

    model,

    test_images,

    test_labels,

    round_number

):

    print(
        "\nGenerating predictions..."
    )


    probabilities = model.predict(

        test_images,

        batch_size=BATCH_SIZE,

        verbose=0

    ).reshape(-1)


    # Fixed threshold for fair comparison.
    threshold = 0.50


    predictions = (

        probabilities >= threshold

    ).astype(
        np.int32
    )


    accuracy = accuracy_score(

        test_labels,

        predictions

    )


    precision = precision_score(

        test_labels,

        predictions,

        zero_division=0

    )


    recall = recall_score(

        test_labels,

        predictions,

        zero_division=0

    )


    f1 = f1_score(

        test_labels,

        predictions,

        zero_division=0

    )


    cm = confusion_matrix(

        test_labels,

        predictions,

        labels=[0, 1]

    )


    print(
        "\n"
        + "=" * 50
    )

    print(
        "FEDERATED ROUND RESULTS"
    )

    print(
        "=" * 50
    )


    print(
        f"Accuracy : "
        f"{accuracy:.4f}"
    )

    print(
        f"Precision: "
        f"{precision:.4f}"
    )

    print(
        f"Recall   : "
        f"{recall:.4f}"
    )

    print(
        f"F1 Score : "
        f"{f1:.4f}"
    )


    print(
        "\nConfusion Matrix:"
    )

    print(
        cm
    )


    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    result_file = (

        RESULTS_DIR
        / "federated_round_results.csv"

    )


    write_header = (
        not result_file.exists()
    )


    with open(

        result_file,

        "a",

        encoding="utf-8"

    ) as file:

        if write_header:

            file.write(

                "round,"
                "accuracy,"
                "precision,"
                "recall,"
                "f1,"
                "tn,"
                "fp,"
                "fn,"
                "tp\n"

            )


        file.write(

            f"{round_number},"
            f"{accuracy:.6f},"
            f"{precision:.6f},"
            f"{recall:.6f},"
            f"{f1:.6f},"
            f"{cm[0,0]},"
            f"{cm[0,1]},"
            f"{cm[1,0]},"
            f"{cm[1,1]}\n"

        )


    return (

        accuracy,
        precision,
        recall,
        f1,
        cm

    )


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Configure hardware
    # --------------------------------------------------------

    configure_device()


    print(
        "=" * 70
    )

    print(
        "FEDERATED VGG16 - FEDAVG"
    )

    print(
        "=" * 70
    )


    # --------------------------------------------------------
    # Load test dataset
    # --------------------------------------------------------

    test_images, test_labels = (
        load_test_dataset()
    )


    # --------------------------------------------------------
    # Create global model
    # --------------------------------------------------------

    print(
        "\nCreating global VGG16 model..."
    )


    global_model = create_model()


    global_weights = (
        global_model.get_weights()
    )


    # --------------------------------------------------------
    # FEDERATED ROUNDS
    # --------------------------------------------------------

    for round_number in range(

        1,

        NUM_ROUNDS + 1

    ):


        print(
            "\n"
            + "=" * 70
        )

        print(
            f"FEDERATED ROUND "
            f"{round_number}"
        )

        print(
            "=" * 70
        )


        client_weights = []

        client_sizes = []


        # ====================================================
        # CLIENT TRAINING
        # ====================================================

        for client_number in range(

            1,

            NUM_CLIENTS + 1

        ):


            print(
                "\n"
                + "-" * 60
            )

            print(
                f"CLIENT "
                f"{client_number}"
            )

            print(
                "-" * 60
            )


            # ------------------------------------------------
            # Get balanced files
            # ------------------------------------------------

            files, labels = (

                create_balanced_client_data(

                    client_number

                )

            )


            # ------------------------------------------------
            # Create dataset
            # ------------------------------------------------

            train_dataset = (

                create_client_dataset(

                    files,

                    labels

                )

            )


            # ------------------------------------------------
            # Create local model
            # ------------------------------------------------

            local_model = create_model()


            # ------------------------------------------------
            # Start from global model
            # ------------------------------------------------

            local_model.set_weights(

                global_weights

            )


            print(
                "\nStarting local training..."
            )


            # ------------------------------------------------
            # Local training
            # ------------------------------------------------

            history = local_model.fit(

                train_dataset,

                epochs=LOCAL_EPOCHS,

                verbose=1

            )


            print(
                f"\nClient "
                f"{client_number} "
                f"training complete."
            )


            # ------------------------------------------------
            # Local training results
            # ------------------------------------------------

            print(
                "\nLOCAL TRAINING RESULTS"
            )


            for metric_name in [

                "loss",

                "accuracy",

                "precision",

                "recall"

            ]:


                if (
                    metric_name
                    in history.history
                ):

                    value = (

                        history
                        .history[
                            metric_name
                        ][-1]

                    )


                    print(

                        f"{metric_name:<12}: "
                        f"{value:.4f}"

                    )


            # ------------------------------------------------
            # Store local weights
            # ------------------------------------------------

            client_weights.append(

                local_model.get_weights()

            )


            # ------------------------------------------------
            # Number of local samples
            # ------------------------------------------------

            client_sizes.append(

                len(files)

            )


            # ------------------------------------------------
            # Free memory
            # ------------------------------------------------

            del local_model

            del train_dataset

            del files

            del labels

            gc.collect()


            tf.keras.backend.clear_session()


        # ====================================================
        # FEDAVG AGGREGATION
        # ====================================================

        print(
            "\n"
            + "=" * 60
        )

        print(
            "FEDAVG AGGREGATION"
        )

        print(
            "=" * 60
        )


        print(
            "\nClient sample sizes:"
        )


        for client_number, size in enumerate(

            client_sizes,

            start=1

        ):

            print(

                f"Client "
                f"{client_number}: "
                f"{size}"

            )


        print(
            f"\nTotal samples: "
            f"{sum(client_sizes)}"
        )


        print(
            "\nAveraging client weights..."
        )


        global_weights = fedavg(

            client_weights,

            client_sizes

        )


        global_model.set_weights(

            global_weights

        )


        print(
            "FedAvg aggregation completed."
        )


        # ====================================================
        # SAVE GLOBAL MODEL
        # ====================================================

        model_path = (

            MODEL_DIR
            /
            f"federated_vgg16_round"
            f"{round_number}.keras"

        )


        global_model.save(

            model_path

        )


        print(
            "\nGlobal model saved:"
        )

        print(
            model_path
        )


        # ====================================================
        # EVALUATION
        # ====================================================

        (

            accuracy,
            precision,
            recall,
            f1,
            cm

        ) = evaluate_global_model(

            global_model,

            test_images,

            test_labels,

            round_number

        )


        print(
            f"\nFEDERATED ROUND "
            f"{round_number} COMPLETE"
        )


        print(
            f"Round {round_number} "
            f"accuracy: "
            f"{accuracy * 100:.2f}%"
        )


    # ========================================================
    # SAVE FINAL MODEL
    # ========================================================

    final_model_path = (

        MODEL_DIR
        /
        "federated_vgg16_final.keras"

    )


    global_model.save(

        final_model_path

    )


    print(
        "\n"
        + "=" * 70
    )

    print(
        "FEDERATED TRAINING COMPLETE"
    )

    print(
        "=" * 70
    )


    print(
        "\nFinal model:"
    )

    print(
        final_model_path
    )


    print(
        "\nResults file:"
    )

    print(

        RESULTS_DIR
        /
        "federated_round_results.csv"

    )


# ============================================================
# PROGRAM ENTRY
# ============================================================

if __name__ == "__main__":

    main()