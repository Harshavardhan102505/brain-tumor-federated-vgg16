import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(
    0,
    str(PROJECT_ROOT / "src")
)

from federated.federated_training import (
    create_model,
    get_client_files,
    create_client_dataset,
    LOCAL_EPOCHS
)

# ============================================================
# CONFIGURATION
# ============================================================

CLIENT_NUMBER = 1
IMG_SIZE = 224


# ============================================================
# LOAD CLIENT DATA
# ============================================================

print("=" * 70)
print("CLIENT 1 LOCAL TRAINING DIAGNOSTIC")
print("=" * 70)

files, labels = get_client_files(
    CLIENT_NUMBER
)

print("\nCreating client dataset...")

dataset = create_client_dataset(
    files,
    labels
)

# ============================================================
# CREATE MODEL
# ============================================================

print("\nCreating VGG16 model...")

model = create_model()

print("\nTrainable parameters:")

print(
    sum(
        tf.keras.backend.count_params(w)
        for w in model.trainable_weights
    )
)

# ============================================================
# TRAIN
# ============================================================

print("\nStarting local training...")

history = model.fit(
    dataset,
    epochs=LOCAL_EPOCHS,
    verbose=1
)

# ============================================================
# PRINT TRAINING RESULTS
# ============================================================

print("\n" + "=" * 70)
print("LOCAL TRAINING RESULTS")
print("=" * 70)

for key, values in history.history.items():

    print(
        f"{key:15s}: "
        f"{values[-1]:.4f}"
    )

# ============================================================
# SAVE CLIENT MODEL
# ============================================================

output_path = (
    PROJECT_ROOT
    / "models"
    / "diagnostic_client1.keras"
)

model.save(output_path)

print(
    f"\nDiagnostic model saved:\n"
    f"{output_path}"
)

print("\n" + "=" * 70)
print("DIAGNOSTIC COMPLETE")
print("=" * 70)