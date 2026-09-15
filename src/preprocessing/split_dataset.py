import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.config import *

# ----------------------------
# Load processed dataset
# ----------------------------
df = pd.read_csv(PROCESSED_DATA_DIR / "processed_dataset.csv")

print("Processed dataset loaded!")

# ----------------------------
# Get unique patients
# ----------------------------
patients = df["PatientNumber"].unique()

print(f"Total Patients: {len(patients)}")

# ----------------------------
# Split patients
# ----------------------------
train_patients, temp_patients = train_test_split(
    patients,
    test_size=0.30,
    random_state=42,
    shuffle=True
)

val_patients, test_patients = train_test_split(
    temp_patients,
    test_size=0.50,
    random_state=42,
    shuffle=True
)

# ----------------------------
# Create datasets
# ----------------------------
train_df = df[df["PatientNumber"].isin(train_patients)]
val_df = df[df["PatientNumber"].isin(val_patients)]
test_df = df[df["PatientNumber"].isin(test_patients)]

# ----------------------------
# Save datasets
# ----------------------------
train_df.to_csv(PROCESSED_DATA_DIR / "train.csv", index=False)
val_df.to_csv(PROCESSED_DATA_DIR / "validation.csv", index=False)
test_df.to_csv(PROCESSED_DATA_DIR / "test.csv", index=False)

print("\nPatient-wise split completed!\n")

print(f"Training Patients   : {len(train_patients)}")
print(f"Validation Patients : {len(val_patients)}")
print(f"Testing Patients    : {len(test_patients)}")

print()

print(f"Training Images     : {len(train_df)}")
print(f"Validation Images   : {len(val_df)}")
print(f"Testing Images      : {len(test_df)}")