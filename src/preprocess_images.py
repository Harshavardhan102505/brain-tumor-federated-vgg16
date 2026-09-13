from pathlib import Path
import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "data" / "preprocessed"

IMAGE_SIZE = (224, 224)


def process_split(split):

    input_split = INPUT_DIR / split
    output_split = OUTPUT_DIR / split

    for class_dir in input_split.iterdir():

        if not class_dir.is_dir():
            continue

        output_class = output_split / class_dir.name
        output_class.mkdir(parents=True, exist_ok=True)

        images = list(class_dir.iterdir())

        print(f"\n{split}/{class_dir.name}: {len(images)} images")

        for i, image_path in enumerate(images):

            try:
                image = cv2.imread(str(image_path))

                if image is None:
                    print(f"Could not read: {image_path}")
                    continue

                # Resize to VGG16 input size
                image = cv2.resize(image, IMAGE_SIZE)

                # Convert BGR → RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # Normalize pixel values to [0,1]
                image = image.astype(np.float32) / 255.0

                # Save as NumPy array
                output_path = output_class / (
                    image_path.stem + ".npy"
                )

                np.save(output_path, image)

                if (i + 1) % 500 == 0:
                    print(f"Processed {i + 1}/{len(images)}")

            except Exception as e:
                print(f"Error processing {image_path}: {e}")


def main():

    print("=" * 60)
    print("BRAIN MRI IMAGE PREPROCESSING")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for split in ["train", "val", "test"]:
        process_split(split)

    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)

    print(f"\nOutput directory:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()