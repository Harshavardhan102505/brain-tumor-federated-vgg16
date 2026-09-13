import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

RESULTS_DIR = PROJECT_ROOT / "results"

OUTPUT_DIR = RESULTS_DIR / "round_comparison"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# YOUR FEDERATED RESULTS
# ============================================================
#
# Accuracy, Precision, Recall and F1 are taken from
# your completed federated rounds.
#
# Loss should be filled from your actual training output.
#
# If your program saved loss automatically, the function
# below can load it.
# ============================================================

round_results = {
    1: {
        "accuracy": 95.42,
        "precision": 97.56,
        "recall": 96.30,
        "f1": 96.92,
        "loss": None
    },

    2: {
        "accuracy": 96.11,
        "precision": 97.76,
        "recall": 97.04,
        "f1": 97.40,
        "loss": None
    },

    3: {
        "accuracy": 96.53,
        "precision": 98.31,
        "recall": 97.04,
        "f1": 97.67,
        "loss": None
    },

    4: {
        "accuracy": 96.94,
        "precision": 98.50,
        "recall": 97.41,
        "f1": 97.95,
        "loss": None
    },

    5: {
        "accuracy": 97.22,
        "precision": 98.51,
        "recall": 97.78,
        "f1": 98.14,
        "loss": None
    }
}


# ============================================================
# OPTIONAL: LOAD SAVED ROUND RESULTS
# ============================================================

def load_saved_results():

    possible_files = [
        RESULTS_DIR / "round_results.json",
        RESULTS_DIR / "federated_results.json",
        RESULTS_DIR / "training_results.json"
    ]

    for file_path in possible_files:

        if file_path.exists():

            try:

                with open(
                    file_path,
                    "r",
                    encoding="utf-8"
                ) as f:

                    saved = json.load(f)

                print(
                    f"Loaded saved results from: {file_path}"
                )

                return saved

            except Exception as e:

                print(
                    f"Could not read {file_path}: {e}"
                )

    return None


saved_results = load_saved_results()


# ============================================================
# MERGE SAVED LOSS VALUES
# ============================================================

if saved_results:

    for round_number in round_results:

        key = str(round_number)

        if key in saved_results:

            saved_round = saved_results[key]

            if isinstance(saved_round, dict):

                if "loss" in saved_round:

                    round_results[
                        round_number
                    ]["loss"] = saved_round["loss"]


# ============================================================
# CREATE DATAFRAME
# ============================================================

data = []

for round_number, result in round_results.items():

    data.append({
        "Round": round_number,
        "Loss": result["loss"],
        "Accuracy (%)": result["accuracy"],
        "Precision (%)": result["precision"],
        "Recall (%)": result["recall"],
        "F1 Score (%)": result["f1"]
    })


df = pd.DataFrame(data)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print()
print("=" * 75)
print("FEDERATED VGG16 ROUND-WISE CLASSIFICATION REPORT")
print("=" * 75)

print()

print(
    df.to_string(
        index=False
    )
)

print()


# ============================================================
# SAVE CSV
# ============================================================

csv_file = (
    OUTPUT_DIR /
    "federated_round_comparison.csv"
)

df.to_csv(
    csv_file,
    index=False
)

print(
    f"CSV saved: {csv_file}"
)


# ============================================================
# SAVE EXCEL
# ============================================================

excel_file = (
    OUTPUT_DIR /
    "federated_round_comparison.xlsx"
)

df.to_excel(
    excel_file,
    index=False
)

print(
    f"Excel saved: {excel_file}"
)


# ============================================================
# BEST ROUND
# ============================================================

best_accuracy_index = df[
    "Accuracy (%)"
].idxmax()

best_accuracy_round = int(
    df.loc[
        best_accuracy_index,
        "Round"
    ]
)

best_accuracy = df.loc[
    best_accuracy_index,
    "Accuracy (%)"
]


best_f1_index = df[
    "F1 Score (%)"
].idxmax()

best_f1_round = int(
    df.loc[
        best_f1_index,
        "Round"
    ]
)

best_f1 = df.loc[
    best_f1_index,
    "F1 Score (%)"
]


best_recall_index = df[
    "Recall (%)"
].idxmax()

best_recall_round = int(
    df.loc[
        best_recall_index,
        "Round"
    ]
)

best_recall = df.loc[
    best_recall_index,
    "Recall (%)"
]


print("=" * 75)
print("BEST RESULTS")
print("=" * 75)

print(
    f"Best Accuracy : Round {best_accuracy_round} "
    f"({best_accuracy:.2f}%)"
)

print(
    f"Best F1 Score : Round {best_f1_round} "
    f"({best_f1:.2f}%)"
)

print(
    f"Best Recall   : Round {best_recall_round} "
    f"({best_recall:.2f}%)"
)

print()


# ============================================================
# GRAPH 1
# ACCURACY
# ============================================================

plt.figure(
    figsize=(9, 5)
)

plt.plot(
    df["Round"],
    df["Accuracy (%)"],
    marker="o",
    linewidth=2
)

plt.xticks(
    df["Round"]
)

plt.xlabel(
    "Federated Round"
)

plt.ylabel(
    "Accuracy (%)"
)

plt.title(
    "Federated VGG16 Accuracy Comparison"
)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

accuracy_graph = (
    OUTPUT_DIR /
    "accuracy_comparison.png"
)

plt.savefig(
    accuracy_graph,
    dpi=200
)

plt.close()


# ============================================================
# GRAPH 2
# LOSS
# ============================================================

if df["Loss"].notna().any():

    loss_df = df.dropna(
        subset=["Loss"]
    )

    plt.figure(
        figsize=(9, 5)
    )

    plt.plot(
        loss_df["Round"],
        loss_df["Loss"],
        marker="o",
        linewidth=2
    )

    plt.xticks(
        df["Round"]
    )

    plt.xlabel(
        "Federated Round"
    )

    plt.ylabel(
        "Loss"
    )

    plt.title(
        "Federated VGG16 Loss Comparison"
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    loss_graph = (
        OUTPUT_DIR /
        "loss_comparison.png"
    )

    plt.savefig(
        loss_graph,
        dpi=200
    )

    plt.close()

    print(
        f"Loss graph saved: {loss_graph}"
    )

else:

    print(
        "WARNING: Loss values were not found."
    )

    print(
        "Loss graph was not generated."
    )


# ============================================================
# GRAPH 3
# PRECISION
# ============================================================

plt.figure(
    figsize=(9, 5)
)

plt.plot(
    df["Round"],
    df["Precision (%)"],
    marker="o",
    linewidth=2
)

plt.xticks(
    df["Round"]
)

plt.xlabel(
    "Federated Round"
)

plt.ylabel(
    "Precision (%)"
)

plt.title(
    "Federated VGG16 Precision Comparison"
)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

precision_graph = (
    OUTPUT_DIR /
    "precision_comparison.png"
)

plt.savefig(
    precision_graph,
    dpi=200
)

plt.close()


# ============================================================
# GRAPH 4
# RECALL
# ============================================================

plt.figure(
    figsize=(9, 5)
)

plt.plot(
    df["Round"],
    df["Recall (%)"],
    marker="o",
    linewidth=2
)

plt.xticks(
    df["Round"]
)

plt.xlabel(
    "Federated Round"
)

plt.ylabel(
    "Recall (%)"
)

plt.title(
    "Federated VGG16 Recall Comparison"
)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

recall_graph = (
    OUTPUT_DIR /
    "recall_comparison.png"
)

plt.savefig(
    recall_graph,
    dpi=200
)

plt.close()


# ============================================================
# GRAPH 5
# F1 SCORE
# ============================================================

plt.figure(
    figsize=(9, 5)
)

plt.plot(
    df["Round"],
    df["F1 Score (%)"],
    marker="o",
    linewidth=2
)

plt.xticks(
    df["Round"]
)

plt.xlabel(
    "Federated Round"
)

plt.ylabel(
    "F1 Score (%)"
)

plt.title(
    "Federated VGG16 F1 Score Comparison"
)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

f1_graph = (
    OUTPUT_DIR /
    "f1_comparison.png"
)

plt.savefig(
    f1_graph,
    dpi=200
)

plt.close()


# ============================================================
# GRAPH 6
# ALL CLASSIFICATION METRICS
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    df["Round"],
    df["Accuracy (%)"],
    marker="o",
    linewidth=2,
    label="Accuracy"
)

plt.plot(
    df["Round"],
    df["Precision (%)"],
    marker="o",
    linewidth=2,
    label="Precision"
)

plt.plot(
    df["Round"],
    df["Recall (%)"],
    marker="o",
    linewidth=2,
    label="Recall"
)

plt.plot(
    df["Round"],
    df["F1 Score (%)"],
    marker="o",
    linewidth=2,
    label="F1 Score"
)

plt.xticks(
    df["Round"]
)

plt.xlabel(
    "Federated Round"
)

plt.ylabel(
    "Score (%)"
)

plt.title(
    "Federated VGG16 Classification Metrics Across Rounds"
)

plt.legend()

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

all_metrics_graph = (
    OUTPUT_DIR /
    "all_metrics_comparison.png"
)

plt.savefig(
    all_metrics_graph,
    dpi=200
)

plt.close()


# ============================================================
# FINAL REPORT TEXT
# ============================================================

report_file = (
    OUTPUT_DIR /
    "federated_classification_report.txt"
)

with open(
    report_file,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "FEDERATED VGG16 CLASSIFICATION REPORT\n"
    )

    f.write(
        "=" * 60 + "\n\n"
    )

    f.write(
        "Round-wise Results\n\n"
    )

    f.write(
        df.to_string(
            index=False
        )
    )

    f.write(
        "\n\n"
    )

    f.write(
        "BEST RESULTS\n"
    )

    f.write(
        "=" * 60 + "\n"
    )

    f.write(
        f"Best Accuracy : "
        f"Round {best_accuracy_round} "
        f"({best_accuracy:.2f}%)\n"
    )

    f.write(
        f"Best F1 Score : "
        f"Round {best_f1_round} "
        f"({best_f1:.2f}%)\n"
    )

    f.write(
        f"Best Recall   : "
        f"Round {best_recall_round} "
        f"({best_recall:.2f}%)\n"
    )


# ============================================================
# COMPLETE
# ============================================================

print("=" * 75)
print("REPORT GENERATION COMPLETE")
print("=" * 75)

print()
print(
    f"Output folder:\n{OUTPUT_DIR}"
)

print()

print(
    "Generated files:"
)

print(
    "1. federated_round_comparison.csv"
)

print(
    "2. federated_round_comparison.xlsx"
)

print(
    "3. federated_classification_report.txt"
)

print(
    "4. accuracy_comparison.png"
)

print(
    "5. precision_comparison.png"
)

print(
    "6. recall_comparison.png"
)

print(
    "7. f1_comparison.png"
)

print(
    "8. all_metrics_comparison.png"
)

if df["Loss"].notna().any():

    print(
        "9. loss_comparison.png"
    )

print()