import pandas as pd
from pathlib import Path
from src.utils.config import *
# Read the diagnosis CSV
df = pd.read_csv(DIAGNOSIS_CSV)

print("Dataset loaded successfully!\n")

print("First 5 Rows")
print(df.head())

print("\nShape of Dataset:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nMissing Values:")
print(df.isnull().sum())

# ----------------------------
# Create Image Paths
# ----------------------------

image_paths = []

for _, row in df.iterrows():
    patient = f"{int(row['PatientNumber']):03d}"   # 49 -> 049
    slice_no = int(row["SliceNumber"])

    image_path = PATIENTS_DIR / patient / "brain" / f"{slice_no}.jpg"

    image_paths.append(str(image_path))

df["Image_Path"] = image_paths

print("\nImage Paths Created Successfully!")
print(df[["PatientNumber", "SliceNumber", "Image_Path"]].head())

# ----------------------------
# Check if image exists
# ----------------------------

df["Image_Exists"] = df["Image_Path"].apply(lambda x: Path(x).exists())

print("\nImage Existence Check:")
print(df["Image_Exists"].value_counts())

# Keep only valid images
df = df[df["Image_Exists"]].copy()

print("\nDataset Shape After Removing Missing Images:")
print(df.shape)

# ----------------------------
# Convert One-Hot Labels to a Single Label
# ----------------------------

label_columns = [
    "Intraventricular",
    "Intraparenchymal",
    "Subarachnoid",
    "Epidural",
    "Subdural",
    "No_Hemorrhage"
]

# Get the label name where the value is 1
df["Label"] = df[label_columns].idxmax(axis=1)

print("\nFirst 10 Labels:")
print(df[["PatientNumber", "SliceNumber", "Label"]].head(10))

# ----------------------------
# Save Processed Dataset
# ----------------------------

# Create the output folder if it doesn't exist
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

# File path to save the processed dataset
processed_csv = PROCESSED_DATA_DIR / "processed_dataset.csv"

# Save the dataframe
df.to_csv(processed_csv, index=False)

print("\nProcessed dataset saved successfully!")
print(f"Location: {processed_csv}")

