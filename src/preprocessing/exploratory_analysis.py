import pandas as pd
import matplotlib.pyplot as plt

from src.utils.config import *

# ----------------------------
# Load dataset
# ----------------------------
df = pd.read_csv(PROCESSED_DATA_DIR / "processed_dataset.csv")

print("Dataset Loaded Successfully!\n")

# ----------------------------
# Class Distribution
# ----------------------------
class_counts = df["Label"].value_counts()

plt.figure(figsize=(10,6))

ax = class_counts.plot(kind="bar")

plt.title("Hemorrhage Class Distribution")
plt.xlabel("Hemorrhage Type")
plt.ylabel("Number of CT Images")

plt.xticks(rotation=30)

# Add numbers above bars
for container in ax.containers:
    ax.bar_label(container)

plt.tight_layout()

plt.savefig(PLOTS_DIR / "class_distribution.png")

plt.show()

print("\nGraph saved successfully!")

# ----------------------------
# Images per Patient
# ----------------------------

patient_counts = df.groupby("PatientNumber").size()

plt.figure(figsize=(12,5))

patient_counts.plot(kind="bar")

plt.title("Number of CT Images per Patient")
plt.xlabel("Patient Number")
plt.ylabel("Number of CT Images")

plt.tight_layout()

plt.savefig(PLOTS_DIR / "images_per_patient.png")

plt.show()