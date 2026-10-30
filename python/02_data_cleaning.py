# ============================================================
# STEP 2: DATA CLEANING AND PREPROCESSING
# ============================================================

import pandas as pd
from pathlib import Path

# ============================================================
# 1. DEFINE PROJECT PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "data" / "clinical_trials_raw_patient2trial_conditions.csv"
OUTPUT_DIR = PROJECT_DIR / "outputs"
OUTPUT_PATH = OUTPUT_DIR / "cleaned_clinical_trials.csv"

# ============================================================
# 2. LOAD RAW DATASET
# ============================================================

df = pd.read_csv(DATA_PATH)

print("\n" + "=" * 60)
print("DATA CLEANING STARTED")
print("=" * 60)

print("\nOriginal Dataset Shape:", df.shape)

# ============================================================
# 3. REMOVE EXACT DUPLICATE ROWS
# ============================================================

duplicate_rows = df.duplicated().sum()
df = df.drop_duplicates()

print("Duplicate Rows Removed:", duplicate_rows)

# ============================================================
# 4. REMOVE DUPLICATE TRIAL IDS, IF ANY
# ============================================================

duplicate_ids = df["nct_id"].duplicated().sum()
df = df.drop_duplicates(subset="nct_id", keep="first")

print("Duplicate Trial IDs Removed:", duplicate_ids)

# ============================================================
# 5. CLEAN TARGET LABELS AND SUMMARY TEXT
# ============================================================

df["source_condition_query"] = (
    df["source_condition_query"].astype("string").str.strip().str.lower()
)

df["brief_summary"] = (
    df["brief_summary"].astype("string").str.strip()
)

# ============================================================
# 6. REMOVE RECORDS WITH INVALID TARGET OR SUMMARY
# ============================================================

before_filtering = len(df)

df = df.dropna(subset=["source_condition_query", "brief_summary"])

df = df[
    df["source_condition_query"].ne("")
    & df["brief_summary"].ne("")
].copy()

print("Invalid Records Removed:", before_filtering - len(df))

# ============================================================
# 7. RESET INDEX
# ============================================================

df = df.reset_index(drop=True)

# ============================================================
# 8. SAVE CLEANED DATASET
# ============================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT_PATH, index=False)

# ============================================================
# 9. DISPLAY CLEANING SUMMARY
# ============================================================

print("\nCleaned Dataset Shape:", df.shape)
print("Remaining Duplicate Rows:", df.duplicated().sum())
print("Remaining Duplicate Trial IDs:", df["nct_id"].duplicated().sum())
print("Missing Target Labels:", df["source_condition_query"].isna().sum())
print("Missing Brief Summaries:", df["brief_summary"].isna().sum())

print("\nTarget Category Counts:")
print(df["source_condition_query"].value_counts())

print("\nCleaned Dataset Saved To:")
print(OUTPUT_PATH)

print("\n" + "=" * 60)
print("DATA CLEANING COMPLETED")
print("=" * 60)
