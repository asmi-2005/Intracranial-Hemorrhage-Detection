"""
Configuration file for the Intracranial Hemorrhage Detection Project
"""

from pathlib import Path

# ----------------------------
# Project Root Directory
# ----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ----------------------------
# Dataset Paths
# ----------------------------
DATASET_DIR = PROJECT_ROOT / "dataset"
PATIENTS_DIR = DATASET_DIR / "Patients_CT"

DIAGNOSIS_CSV = DATASET_DIR / "hemorrhage_diagnosis.csv"
DEMOGRAPHICS_CSV = DATASET_DIR / "patient_demographics.csv"

# ----------------------------
# Output Directories
# ----------------------------
OUTPUT_DIR = PROJECT_ROOT / "outputs"

PROCESSED_DATA_DIR = OUTPUT_DIR / "processed_data"
MODELS_DIR = OUTPUT_DIR / "models"
PLOTS_DIR = OUTPUT_DIR / "plots"
REPORTS_DIR = OUTPUT_DIR / "reports"

# ----------------------------
# Training Configuration
# ----------------------------
IMAGE_SIZE = 224
BATCH_SIZE = 16
NUM_CLASSES = 6
RANDOM_SEED = 42