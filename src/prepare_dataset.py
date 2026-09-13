from pathlib import Path
import shutil
import random
import csv

# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

RANDOM_SEED = 42

# Original dataset classes
TUMOR_CLASSES = {
    "glioma",
    "meningioma",
    "pituitary"
}

NO_TUMOR_CLASSES = {
    "notumor"
}


# ============================================================
# FUNCTIONS
# ============================================================

def collect_images():
    """Collect all images from Training and Testing folders."""

    images = []

    for split_folder in ["Training", "Testing"]:

        split_path = RAW_DIR / split_folder

        if not split_path.exists():
            print(f"ERROR: Missing folder: {split_path}")
            continue

        for class_folder in split_path.iterdir():

            if not class_folder.is_dir():
                continue

            original_class = class_folder.name.lower()

            if original_class in TUMOR_CLASSES:
                binary_class = "Tumor"

            elif original_class in NO_TUMOR_CLASSES:
                binary_class = "No Tumor"

            else:
                print(f"WARNING: Unknown class: {class_folder.name}")
                continue

            for image_path in class_folder.iterdir():

                if image_path.suffix.lower() in IMAGE_EXTENSIONS:

                    images.append({
                        "source_path": image_path,
                        "original_class": original_class,
                        "class_name": binary_class
                    })

    return images


def create_split(images):

    """Create stratified 80/10/10 train/validation/test split."""

    random.seed(RANDOM_SEED)

    tumor_images = [
        image for image in images
        if image["class_name"] == "Tumor"
    ]

    no_tumor_images = [
        image for image in images
        if image["class_name"] == "No Tumor"
    ]

    random.shuffle(tumor_images)
    random.shuffle(no_tumor_images)

    def split_class(class_images):

        total = len(class_images)

        train_end = int(total * 0.80)
        val_end = int(total * 0.90)

        return (
            class_images[:train_end],
            class_images[train_end:val_end],
            class_images[val_end:]
        )

    tumor_train, tumor_val, tumor_test = split_class(tumor_images)

    no_train, no_val, no_test = split_class(no_tumor_images)

    train = tumor_train + no_train
    val = tumor_val + no_val
    test = tumor_test + no_test

    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)

    return train, val, test

def copy_images(images, split_name):

    """Copy images into processed dataset."""

    for image in images:

        destination_dir = (
            PROCESSED_DIR
            / split_name
            / image["class_name"]
        )

        destination_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        destination = destination_dir / image["source_path"].name

        # Avoid duplicate filename collisions
        counter = 1

        while destination.exists():

            destination = (
                destination_dir
                / f"{image['source_path'].stem}_{counter}"
                f"{image['source_path'].suffix}"
            )

            counter += 1

        shutil.copy2(
            image["source_path"],
            destination
        )


def save_metadata(images, split_name, writer):

    """Write metadata information."""

    for image in images:

        writer.writerow([
            str(image["source_path"]),
            image["original_class"],
            image["class_name"],
            split_name
        ])


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("BRAIN TUMOR DATASET PREPARATION")
    print("=" * 60)

    images = collect_images()

    if not images:
        print("\nERROR: No images found.")
        print("Check that your dataset is inside:")
        print(RAW_DIR)
        return

    print(f"\nTotal images found: {len(images)}")

    tumor_count = sum(
        1 for image in images
        if image["class_name"] == "Tumor"
    )

    no_tumor_count = sum(
        1 for image in images
        if image["class_name"] == "No Tumor"
    )

    print(f"Tumor images: {tumor_count}")
    print(f"No Tumor images: {no_tumor_count}")

    # Remove previous processed dataset
    if PROCESSED_DIR.exists():
        shutil.rmtree(PROCESSED_DIR)

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    train, val, test = create_split(images)

    print("\nDataset split:")
    print(f"Training:   {len(train)}")
    print(f"Validation: {len(val)}")
    print(f"Testing:    {len(test)}")

    # Copy images
    print("\nCopying training images...")
    copy_images(train, "train")

    print("Copying validation images...")
    copy_images(val, "val")

    print("Copying testing images...")
    copy_images(test, "test")

    # Metadata
    metadata_path = PROCESSED_DIR / "metadata.csv"

    with open(
        metadata_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "source_path",
            "original_class",
            "class_name",
            "split"
        ])

        save_metadata(train, "train", writer)
        save_metadata(val, "val", writer)
        save_metadata(test, "test", writer)

    print("\n" + "=" * 60)
    print("DATASET PREPARATION COMPLETE")
    print("=" * 60)

    print(f"\nProcessed dataset:")
    print(PROCESSED_DIR)

    print(f"\nMetadata:")
    print(metadata_path)


if __name__ == "__main__":
    main()