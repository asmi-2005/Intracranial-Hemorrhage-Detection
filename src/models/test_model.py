import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
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

MODEL_PATH = "outputs/models/efficientnetv2_best.pth"

# CHANGE THIS TO YOUR TEST DATASET LOCATION
TEST_DIR = r"C:\Users\ASMI\OneDrive\Desktop\Intracranial_Hemorrhage_Detection\dataset\test"

BATCH_SIZE = 32


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
# LOAD TEST DATASET
# ============================================================

print("=" * 70)
print("INTRACRANIAL HEMORRHAGE - CONFUSION MATRIX")
print("=" * 70)

print(f"\nUsing device: {DEVICE}")

if not os.path.exists(TEST_DIR):
    print("\nERROR: Test dataset not found!")
    print(f"Expected location:\n{TEST_DIR}")
    exit()

test_dataset = datasets.ImageFolder(
    root=TEST_DIR,
    transform=transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print(f"\nTotal test images: {len(test_dataset)}")

print("\nDataset classes:")
print(test_dataset.class_to_idx)


# ============================================================
# LOAD MODEL
# ============================================================

model = HemorrhageClassifier(num_classes=6).to(DEVICE)

if not os.path.exists(MODEL_PATH):
    print("\nERROR: Model file not found!")
    print(f"Expected model at:\n{MODEL_PATH}")
    exit()

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=False
)

# Handle different checkpoint formats
if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:

    model.load_state_dict(
        checkpoint["state_dict"]
    )

else:

    model.load_state_dict(checkpoint)


model.eval()

print("\nModel loaded successfully!")


# ============================================================
# RUN PREDICTIONS
# ============================================================

all_predictions = []
all_labels = []

print("\nRunning predictions...")

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(DEVICE)

        outputs = model(images)

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_labels.extend(
            labels.numpy()
        )


all_predictions = np.array(all_predictions)
all_labels = np.array(all_labels)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    all_labels,
    all_predictions,
    labels=range(len(CLASS_NAMES))
)

print("\nConfusion Matrix:")
print(cm)


# ============================================================
# DISPLAY CONFUSION MATRIX
# ============================================================

plt.figure(figsize=(10, 8))

plt.imshow(cm, interpolation="nearest")

plt.title(
    "Confusion Matrix - Intracranial Hemorrhage Classification",
    fontsize=14
)

plt.colorbar()

tick_marks = np.arange(len(CLASS_NAMES))

plt.xticks(
    tick_marks,
    CLASS_NAMES,
    rotation=45,
    ha="right"
)

plt.yticks(
    tick_marks,
    CLASS_NAMES
)

plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")


# Add numbers inside the matrix
threshold = cm.max() / 2

for i in range(cm.shape[0]):

    for j in range(cm.shape[1]):

        plt.text(
            j,
            i,
            str(cm[i, j]),
            horizontalalignment="center",
            verticalalignment="center",
            color="white" if cm[i, j] > threshold else "black",
            fontsize=11
        )


plt.tight_layout()

# Save image
os.makedirs("outputs/evaluation", exist_ok=True)

plt.savefig(
    "outputs/evaluation/confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n")
print("=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

report = classification_report(
    all_labels,
    all_predictions,
    labels=range(len(CLASS_NAMES)),
    target_names=CLASS_NAMES,
    digits=4,
    zero_division=0
)

print(report)


# ============================================================
# SAVE CLASSIFICATION REPORT
# ============================================================

with open(
    "outputs/evaluation/classification_report.txt",
    "w"
) as f:

    f.write(report)


print("\nEvaluation completed successfully!")

print(
    "\nConfusion matrix saved to:"
    "\noutputs/evaluation/confusion_matrix.png"
)

print(
    "\nClassification report saved to:"
    "\noutputs/evaluation/classification_report.txt"
)