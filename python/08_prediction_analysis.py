
# ============================================================
# PREDICTION ANALYSIS: CLINICAL TRIAL DISEASE CLASSIFICATION
# ============================================================

import os
import joblib

# ============================================================
# 1. PATH CONFIGURATION
# ============================================================

BASE_DIR = r"C:\Users\acer\Documents\Clinical_Trial_Classification"
MODEL_DIR = os.path.join(BASE_DIR, "models")

# ============================================================
# 2. LOAD SAVED COMPONENTS
# ============================================================

try:
    model = joblib.load(
        os.path.join(MODEL_DIR, "logistic_regression_model.pkl")
    )

    vectorizer = joblib.load(
        os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")
    )

    label_encoder = joblib.load(
        os.path.join(MODEL_DIR, "label_encoder.pkl")
    )

    print("Logistic Regression model loaded successfully.")

except FileNotFoundError as error:
    print(f"Error: Required model file not found: {error.filename}")
    raise SystemExit(1)

except Exception as error:
    print(f"Error loading saved components: {error}")
    raise SystemExit(1)

# ============================================================
# 3. GET CLINICAL TRIAL SUMMARY
# ============================================================

summary = input(
    "\nEnter the clinical trial summary: "
).strip()

if not summary:
    print("Error: Please enter a valid clinical trial summary.")
    raise SystemExit(1)

# ============================================================
# 4. TRANSFORM TEXT
# ============================================================

try:
    # Match the basic lowercase preprocessing used in prediction.
    processed_summary = summary.lower()

    X_input = vectorizer.transform([processed_summary])

    # ========================================================
    # 5. PREDICT DISEASE CATEGORY
    # ========================================================

    prediction = model.predict(X_input)

    disease_category = label_encoder.inverse_transform(
        prediction
    )[0]

    # ========================================================
    # 6. DISPLAY PREDICTION
    # ========================================================

    print("\n" + "=" * 50)
    print("CLINICAL TRIAL DISEASE PREDICTION")
    print("=" * 50)

    print(f"\nInput Summary:\n{summary}")
    print(f"\nPredicted Disease Category: {disease_category}")

    print("\nPrediction completed successfully.")

except Exception as error:
    print(f"\nError during prediction: {error}")
    raise SystemExit(1)
