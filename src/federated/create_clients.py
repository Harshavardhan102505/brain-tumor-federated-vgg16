from pathlib import Path
import shutil
import random


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Use the SAME preprocessed dataset used by the successful VGG16
# training.
SOURCE_DIR = (
    PROJECT_ROOT /
    "data" /
    "preprocessed" /
    "train"
)

CLIENT_DIR = (
    PROJECT_ROOT /
    "data" /
    "federated_clients"
)

NUM_CLIENTS = 4

SEED = 42

random.seed(SEED)


# ============================================================
# CREATE CLIENT DIRECTORIES
# ============================================================

def create_directories():

    for client_number in range(
        1,
        NUM_CLIENTS + 1
    ):

        client_path = (
            CLIENT_DIR /
            f"client_{client_number}"
        )

        tumor_dir = (
            client_path /
            "Tumor"
        )

        no_tumor_dir = (
            client_path /
            "No Tumor"
        )

        tumor_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        no_tumor_dir.mkdir(
            parents=True,
            exist_ok=True
        )


# ============================================================
# CLEAR OLD CLIENT DATA
# ============================================================

def clear_old_client_data():

    print(
        "\nClearing old federated client data..."
    )

    for client_number in range(
        1,
        NUM_CLIENTS + 1
    ):

        client_path = (
            CLIENT_DIR /
            f"client_{client_number}"
        )

        if client_path.exists():

            for class_name in [
                "Tumor",
                "No Tumor"
            ]:

                class_path = (
                    client_path /
                    class_name
                )

                if class_path.exists():

                    for file in class_path.iterdir():

                        if file.is_file():

                            file.unlink()

                        elif file.is_dir():

                            shutil.rmtree(file)

    print(
        "Old client data removed."
    )


# ============================================================
# DISTRIBUTE FILES
# ============================================================

def distribute_class(class_name):

    source_class = (
        SOURCE_DIR /
        class_name
    )

    if not source_class.exists():

        raise FileNotFoundError(
            f"Source directory not found: "
            f"{source_class}"
        )

    # Only use preprocessed NumPy files
    files = list(
        source_class.glob("*.npy")
    )

    if len(files) == 0:

        raise ValueError(
            f"No .npy files found in: "
            f"{source_class}"
        )

    random.shuffle(files)

    total = len(files)

    print(
        f"\n{class_name}: {total} images"
    )

    # Distribute equally among clients
    for index, file in enumerate(files):

        client_number = (
            index % NUM_CLIENTS
        ) + 1

        destination = (
            CLIENT_DIR /
            f"client_{client_number}" /
            class_name /
            file.name
        )

        shutil.copy2(
            file,
            destination
        )

    print(
        f"Distributed across "
        f"{NUM_CLIENTS} clients."
    )


# ============================================================
# VERIFY CLIENTS
# ============================================================

def verify_clients():

    print(
        "\n" + "=" * 60
    )

    print(
        "CLIENT DATA DISTRIBUTION"
    )

    print(
        "=" * 60
    )

    total_all_clients = 0

    for client_number in range(
        1,
        NUM_CLIENTS + 1
    ):

        client_path = (
            CLIENT_DIR /
            f"client_{client_number}"
        )

        tumor_count = len(
            list(
                (
                    client_path /
                    "Tumor"
                ).glob("*.npy")
            )
        )

        no_tumor_count = len(
            list(
                (
                    client_path /
                    "No Tumor"
                ).glob("*.npy")
            )
        )

        total = (
            tumor_count +
            no_tumor_count
        )

        total_all_clients += total

        print(
            f"\nClient {client_number}"
        )

        print(
            f"  Tumor    : {tumor_count}"
        )

        print(
            f"  No Tumor : {no_tumor_count}"
        )

        print(
            f"  Total    : {total}"
        )

    print(
        "\n" + "-" * 60
    )

    print(
        f"Total client samples: "
        f"{total_all_clients}"
    )

    print(
        "-" * 60
    )


# ============================================================
# VERIFY ONLY NPY FILES
# ============================================================

def verify_file_types():

    print(
        "\nChecking client file types..."
    )

    invalid_files = []

    for client_number in range(
        1,
        NUM_CLIENTS + 1
    ):

        client_path = (
            CLIENT_DIR /
            f"client_{client_number}"
        )

        for class_name in [
            "Tumor",
            "No Tumor"
        ]:

            class_path = (
                client_path /
                class_name
            )

            for file in class_path.iterdir():

                if file.is_file() and file.suffix.lower() != ".npy":

                    invalid_files.append(
                        file
                    )

    if invalid_files:

        print(
            "\nWARNING: Non-NPY files found:"
        )

        for file in invalid_files:

            print(
                f"  {file}"
            )

    else:

        print(
            "All client files are .npy"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=" * 60
    )

    print(
        "FEDERATED CLIENT CREATION"
    )

    print(
        "=" * 60
    )

    print(
        f"\nNumber of clients: "
        f"{NUM_CLIENTS}"
    )

    print(
        f"\nSource:"
    )

    print(
        f"{SOURCE_DIR}"
    )

    print(
        f"\nDestination:"
    )

    print(
        f"{CLIENT_DIR}"
    )

    # --------------------------------------------------------
    # Check source
    # --------------------------------------------------------

    if not SOURCE_DIR.exists():

        raise FileNotFoundError(
            f"\nPreprocessed training "
            f"directory not found:\n"
            f"{SOURCE_DIR}"
        )

    # --------------------------------------------------------
    # Clear old JPG client data
    # --------------------------------------------------------

    clear_old_client_data()

    # --------------------------------------------------------
    # Create directories
    # --------------------------------------------------------

    create_directories()

    # --------------------------------------------------------
    # Distribute classes
    # --------------------------------------------------------

    distribute_class(
        "Tumor"
    )

    distribute_class(
        "No Tumor"
    )

    # --------------------------------------------------------
    # Verify distribution
    # --------------------------------------------------------

    verify_clients()

    # --------------------------------------------------------
    # Verify file types
    # --------------------------------------------------------

    verify_file_types()

    print(
        "\n" + "=" * 60
    )

    print(
        "CLIENT CREATION COMPLETE"
    )

    print(
        "=" * 60
    )


if __name__ == "__main__":
    main()