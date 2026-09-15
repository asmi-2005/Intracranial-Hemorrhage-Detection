import os
import sys
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from PIL import Image
from torchvision import transforms
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score
)

# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")
)

sys.path.append(PROJECT_ROOT)

from src.models.model import HemorrhageClassifier


# ============================================================
# CONFIGURATION
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = [
    "No_Hemorrhage",
    "Epidural",
    "Intraparenchymal",
    "Subdural",
    "Intraventricular",
    "Subarachnoid"
]

NUM_CLASSES = len(CLASS_NAMES)

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "models",
    "efficientnetv2_best.pth"
)

TEST_CSV = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "processed_data",
    "test.csv"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "plots"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 70)
print("INTRACRANIAL HEMORRHAGE - MODEL EVALUATION")
print("=" * 70)

print(f"\nDevice: {DEVICE}")
print(f"Model: {MODEL_PATH}")
print(f"Test CSV: {TEST_CSV}")

if not os.path.exists(MODEL_PATH):
    print("\nERROR: Model file not found!")
    print(MODEL_PATH)
    sys.exit()

if not os.path.exists(TEST_CSV):
    print("\nERROR: test.csv not found!")
    print(TEST_CSV)
    sys.exit()


model = HemorrhageClassifier(
    num_classes=NUM_CLASSES
).to(DEVICE)


checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=False
)

# Handle different checkpoint formats
if isinstance(checkpoint, dict):

    if "model_state_dict" in checkpoint:
        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

    elif "state_dict" in checkpoint:
        model.load_state_dict(
            checkpoint["state_dict"]
        )

    else:
        model.load_state_dict(checkpoint)

else:
    model.load_state_dict(checkpoint)


model.eval()

print("\nModel loaded successfully!")


# ============================================================
# LOAD TEST DATA
# ============================================================

df = pd.read_csv(TEST_CSV)

print(f"\nTotal test records: {len(df)}")

print("\nCSV columns:")
print(df.columns.tolist())


# ============================================================
# FIND IMAGE PATH COLUMN
# ============================================================

possible_path_columns = [
    "Image_Path",
    "image_path",
    "ImagePath",
    "path",
    "Path"
]

image_column = None

for column in possible_path_columns:
    if column in df.columns:
        image_column = column
        break

if image_column is None:
    print("\nERROR: Image path column not found!")
    sys.exit()


# ============================================================
# FIND LABEL COLUMN
# ============================================================

possible_label_columns = [
    "Label",
    "label",
    "Class",
    "class",
    "Target",
    "target"
]

label_column = None

for column in possible_label_columns:
    if column in df.columns:
        label_column = column
        break


# ============================================================
# IF LABEL COLUMN DOES NOT EXIST
# CREATE LABEL FROM ONE-HOT COLUMNS
# ============================================================

if label_column is None:

    print("\nLabel column not found.")
    print("Creating labels from class columns...")

    available_classes = [
        c for c in CLASS_NAMES
        if c in df.columns
    ]

    if len(available_classes) != NUM_CLASSES:

        print("\nERROR: Could not find all class columns.")
        print("Available columns:")
        print(df.columns.tolist())
        sys.exit()

    def get_label(row):

        for class_name in CLASS_NAMES:

            if class_name in row.index:

                try:
                    if int(row[class_name]) == 1:
                        return class_name
                except:
                    pass

        return None

    df["Actual_Label"] = df.apply(
        get_label,
        axis=1
    )

    label_column = "Actual_Label"


# ============================================================
# LABEL CONVERSION
# ============================================================

label_to_index = {
    name: index
    for index, name in enumerate(CLASS_NAMES)
}

index_to_label = {
    index: name
    for index, name in enumerate(CLASS_NAMES)
}


# ============================================================
# PREDICTION
# ============================================================

y_true = []
y_pred = []

skipped = 0

print("\nStarting evaluation...")
print("-" * 70)


for i, row in df.iterrows():

    image_path = str(row[image_column]).strip()

    actual_label = str(row[label_column]).strip()

    # --------------------------------------------------------
    # Fix Windows path if necessary
    # --------------------------------------------------------

    if not os.path.exists(image_path):

        # Try relative to project root
        alternative_path = os.path.join(
            PROJECT_ROOT,
            image_path
        )

        if os.path.exists(alternative_path):
            image_path = alternative_path

        else:
            print(
                f"Skipping image {i + 1}: "
                f"{image_path}"
            )

            skipped += 1
            continue

    # --------------------------------------------------------
    # Check actual label
    # --------------------------------------------------------

    if actual_label not in label_to_index:

        print(
            f"Skipping image {i + 1}: "
            f"Unknown label = {actual_label}"
        )

        skipped += 1
        continue

    # --------------------------------------------------------
    # Load image
    # --------------------------------------------------------

    try:

        image = Image.open(
            image_path
        ).convert("RGB")

        image_tensor = transform(
            image
        )

        image_tensor = image_tensor.unsqueeze(
            0
        ).to(DEVICE)

    except Exception as e:

        print(
            f"Could not read image: "
            f"{image_path}"
        )

        print(f"Error: {e}")

        skipped += 1
        continue

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    with torch.no_grad():

        output = model(
            image_tensor
        )

        prediction = torch.argmax(
            output,
            dim=1
        ).item()

    predicted_label = index_to_label[
        prediction
    ]

    # --------------------------------------------------------
    # Store results
    # --------------------------------------------------------

    y_true.append(
        actual_label
    )

    y_pred.append(
        predicted_label
    )

    # Progress
    if (i + 1) % 50 == 0:

        print(
            f"Processed: {i + 1}/{len(df)}"
        )


# ============================================================
# CHECK RESULTS
# ============================================================

if len(y_true) == 0:

    print("\nERROR: No images were successfully evaluated.")
    sys.exit()


print("\n" + "=" * 70)
print("EVALUATION COMPLETED")
print("=" * 70)

print(f"\nImages evaluated : {len(y_true)}")
print(f"Images skipped   : {skipped}")


# ============================================================
# ACCURACY
# ============================================================

accuracy = accuracy_score(
    y_true,
    y_pred
)

print(
    f"\nTest Accuracy: {accuracy * 100:.2f}%"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:")
print("-" * 70)

report = classification_report(
    y_true,
    y_pred,
    labels=CLASS_NAMES,
    target_names=CLASS_NAMES,
    zero_division=0
)

print(report)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=CLASS_NAMES
)


print("\nConfusion Matrix:")
print(cm)


# ============================================================
# SAVE CONFUSION MATRIX
# ============================================================

plt.figure(
    figsize=(10, 8)
)

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=CLASS_NAMES,
    yticklabels=CLASS_NAMES
)

plt.title(
    "Confusion Matrix - Intracranial Hemorrhage Classification",
    fontsize=14
)

plt.xlabel(
    "Predicted Class",
    fontsize=12
)

plt.ylabel(
    "Actual Class",
    fontsize=12
)

plt.xticks(
    rotation=45,
    ha="right"
)

plt.yticks(
    rotation=0
)

plt.tight_layout()


CONFUSION_MATRIX_PATH = os.path.join(
    OUTPUT_DIR,
    "confusion_matrix.png"
)

plt.savefig(
    CONFUSION_MATRIX_PATH,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print(
    f"\nConfusion matrix saved to:"
)
print(
    CONFUSION_MATRIX_PATH
)


# ============================================================
# SAVE CLASSIFICATION REPORT
# ============================================================

report_dict = classification_report(
    y_true,
    y_pred,
    labels=CLASS_NAMES,
    target_names=CLASS_NAMES,
    output_dict=True,
    zero_division=0
)

report_df = pd.DataFrame(
    report_dict
).transpose()

REPORT_PATH = os.path.join(
    OUTPUT_DIR,
    "classification_report.csv"
)

report_df.to_csv(
    REPORT_PATH
)

print(
    f"\nClassification report saved to:"
)
print(
    REPORT_PATH
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("EVALUATION SUMMARY")
print("=" * 70)

print(
    f"Test Accuracy : {accuracy * 100:.2f}%"
)

print(
    f"Images        : {len(y_true)}"
)

print(
    f"Skipped       : {skipped}"
)

print(
    "\nGenerated files:"
)

print(
    "1. outputs/plots/confusion_matrix.png"
)

print(
    "2. outputs/plots/classification_report.csv"
)

print("=" * 70)