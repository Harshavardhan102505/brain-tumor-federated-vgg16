# Privacy-Aware Brain Tumor Detection Using Federated Transfer Learning

This starter repository follows the technology stack and pipeline described in the supplied interim report:

1. MRI dataset acquisition
2. Resize/normalize/preprocess
3. Custom CNN baseline
4. VGG16 transfer learning
5. Federated Learning with FedAvg
6. Grad-CAM explainability
7. Flask web application
8. Evaluation and integration

## First implementation phase

Put the Kaggle dataset into:

data/raw/yes/
data/raw/no/

Then run:

```bash
python src/prepare_dataset.py
```

This creates an 80/10/10 train/validation/test split and a metadata CSV.

Next, run:

```bash
python src/inspect_dataset.py
```

The script reports class counts, image dimensions, and creates a sample visualization in `results/`.

## Important

This is an academic prototype, not a clinical diagnostic system. Do not use predictions for medical decisions.
