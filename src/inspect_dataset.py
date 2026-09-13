from pathlib import Path
import cv2
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
META = ROOT / "data" / "processed" / "metadata.csv"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

def main():
    if not META.exists():
        raise SystemExit("Run prepare_dataset.py first.")

    df = pd.read_csv(META)
    print("\nTotal images:", len(df))
    print("\nClass distribution:")
    print(df["class_name"].value_counts())
    print("\nSplit distribution:")
    print(pd.crosstab(df["split"], df["class_name"]))

    samples = []
    for _, row in df.sample(min(8, len(df)), random_state=42).iterrows():
        p = ROOT / row["source_path"]
        img = cv2.imread(str(p))
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            samples.append((img, row["class_name"]))

    if samples:
        fig = plt.figure(figsize=(12, 8))
        for i, (img, label) in enumerate(samples):
            ax = fig.add_subplot(2, 4, i + 1)
            ax.imshow(img)
            ax.set_title(label)
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(RESULTS / "dataset_samples.png", dpi=150)
        plt.close(fig)
        print(f"\nSample visualization saved to {RESULTS / 'dataset_samples.png'}")

if __name__ == "__main__":
    main()
