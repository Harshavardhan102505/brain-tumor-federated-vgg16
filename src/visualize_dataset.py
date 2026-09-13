from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "preprocessed"
RESULTS_DIR = PROJECT_ROOT / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CLASSES = ["Tumor", "No Tumor"]
SPLITS = ["train", "val", "test"]


def count_images():
    counts = {}

    for split in SPLITS:
        counts[split] = {}

        for class_name in CLASSES:
            folder = DATA_DIR / split / class_name

            if folder.exists():
                counts[split][class_name] = len(
                    list(folder.glob("*.npy"))
                )
            else:
                counts[split][class_name] = 0

    return counts


def create_class_distribution(counts):

    tumor_counts = [
        counts[split]["Tumor"]
        for split in SPLITS
    ]

    no_tumor_counts = [
        counts[split]["No Tumor"]
        for split in SPLITS
    ]

    x = np.arange(len(SPLITS))
    width = 0.35

    plt.figure(figsize=(8, 6))

    plt.bar(
        x - width / 2,
        tumor_counts,
        width,
        label="Tumor"
    )

    plt.bar(
        x + width / 2,
        no_tumor_counts,
        width,
        label="No Tumor"
    )

    plt.xticks(x, ["Training", "Validation", "Testing"])

    plt.xlabel("Dataset Split")
    plt.ylabel("Number of Images")
    plt.title("Tumor vs No Tumor Distribution")

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "class_distribution.png",
        dpi=300
    )

    plt.close()


def create_sample_grid():

    samples = []

    for class_name in CLASSES:

        folder = DATA_DIR / "train" / class_name

        files = list(folder.glob("*.npy"))

        for file in files[:5]:

            image = np.load(file)

            samples.append(
                (image, class_name)
            )

    plt.figure(figsize=(12, 5))

    for i, (image, label) in enumerate(samples):

        plt.subplot(2, 5, i + 1)

        plt.imshow(image)

        plt.title(label)

        plt.axis("off")

    plt.suptitle("Sample Brain MRI Images")

    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "sample_mri_grid.png",
        dpi=300
    )

    plt.close()


def create_preprocessing_example():

    tumor_folder = DATA_DIR / "train" / "Tumor"

    files = list(tumor_folder.glob("*.npy"))

    if not files:
        print("No tumor images found.")
        return

    image = np.load(files[0])

    print("\nPreprocessing verification:")

    print("Image shape:", image.shape)

    print("Data type:", image.dtype)

    print("Minimum pixel value:", image.min())

    print("Maximum pixel value:", image.max())

    plt.figure(figsize=(6, 6))

    plt.imshow(image)

    plt.title(
        "Preprocessed MRI (224 × 224)"
    )

    plt.axis("off")

    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "preprocessed_example.png",
        dpi=300
    )

    plt.close()


def main():

    print("=" * 60)
    print("DATASET VISUALIZATION")
    print("=" * 60)

    counts = count_images()

    print("\nDataset counts:")

    for split in SPLITS:

        print(f"\n{split.upper()}")

        for class_name in CLASSES:

            print(
                f"{class_name}: "
                f"{counts[split][class_name]}"
            )

    create_class_distribution(counts)

    print("\nCreated:")
    print("results/class_distribution.png")

    create_sample_grid()

    print("results/sample_mri_grid.png")

    create_preprocessing_example()

    print("results/preprocessed_example.png")

    print("\n" + "=" * 60)
    print("VISUALIZATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()