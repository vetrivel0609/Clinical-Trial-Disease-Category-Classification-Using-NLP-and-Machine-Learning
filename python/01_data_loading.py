# ============================================================
# STEP 1: DATA LOADING AND INITIAL INSPECTION
# ============================================================

import pandas as pd
from pathlib import Path

# ============================================================
# 1. DEFINE PROJECT PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "data" / "clinical_trials_raw_patient2trial_conditions.csv"

# ============================================================
# 2. LOAD DATASET
# ============================================================

df = pd.read_csv(DATA_PATH)

# ============================================================
# 3. DISPLAY BASIC DATASET INFORMATION
# ============================================================

print("\n" + "=" * 60)
print("DATASET LOADED SUCCESSFULLY")
print("=" * 60)

print(f"\nDataset Shape: {df.shape}")
print(f"Total Rows: {df.shape[0]}")
print(f"Total Columns: {df.shape[1]}")

# ============================================================
# 4. DISPLAY COLUMN NAMES
# ============================================================

print("\nColumn Names:")
print(df.columns.tolist())

# ============================================================
# 5. DISPLAY FIRST FIVE RECORDS
# ============================================================

print("\nFirst Five Records:")
print(df.head())

# ============================================================
# 6. DISPLAY DATA TYPES AND NON-NULL COUNTS
# ============================================================

print("\nDataset Information:")
df.info()

# ============================================================
# 7. CHECK MISSING VALUES
# ============================================================

print("\nMissing Values by Column:")
print(df.isnull().sum())

print("\n" + "=" * 60)
print("STEP 1 COMPLETED")
print("=" * 60)