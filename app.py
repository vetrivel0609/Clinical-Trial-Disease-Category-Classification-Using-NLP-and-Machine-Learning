# ============================================================
# CLINICAL TRIAL DISEASE CATEGORY CLASSIFICATION
# STREAMLIT WEB APPLICATION
# ============================================================

import os
import joblib
import streamlit as st

# ============================================================
# 1. PATH CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

# ============================================================
# 2. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Clinical Trial Disease Classifier",
    page_icon="🧬",
    layout="wide"
)

# ============================================================
# 3. LOAD SAVED COMPONENTS
# ============================================================

@st.cache_resource
def load_components():
    model = joblib.load(
        os.path.join(MODEL_DIR, "logistic_regression_model.pkl")
    )

    vectorizer = joblib.load(
        os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")
    )

    label_encoder = joblib.load(
        os.path.join(MODEL_DIR, "label_encoder.pkl")
    )

    return model, vectorizer, label_encoder


# ============================================================
# 4. APPLICATION HEADER
# ============================================================

st.title("🧬 Clinical Trial Disease Classification")

st.write(
    "Classify a clinical trial into a disease category "
    "using Natural Language Processing (NLP) and "
    "Machine Learning."
)

st.divider()

# ============================================================
# 5. LOAD MODEL
# ============================================================

try:
    model, vectorizer, label_encoder = load_components()
    st.success("Logistic Regression model loaded successfully.")

except Exception as error:
    st.error(f"Unable to load the saved model components: {error}")
    st.stop()

# ============================================================
# 6. USER INPUT
# ============================================================

st.subheader("Enter Clinical Trial Details")

summary = st.text_area(
    "Clinical Trial Summary",
    placeholder=(
        "Example: This clinical trial evaluates a new treatment "
        "for patients with type 2 diabetes..."
    ),
    height=180,
    help="Enter the clinical trial summary you want to classify."
)

# ============================================================
# 7. PREDICTION
# ============================================================

if st.button("Predict Disease Category", type="primary"):

    if not summary.strip():
        st.warning("Please enter a clinical trial summary.")

    else:
        try:
            processed_summary = summary.strip().lower()

            X_input = vectorizer.transform([processed_summary])

            prediction = model.predict(X_input)

            disease_category = label_encoder.inverse_transform(
                prediction
            )[0]

            st.divider()
            st.subheader("Prediction Result")

            st.success(
                f"Predicted Disease Category: {disease_category}"
            )

        except Exception as error:
            st.error(f"Prediction failed: {error}")

# ============================================================
# 8. PROJECT INFORMATION
# ============================================================

st.divider()

with st.expander("About This Project"):
    st.markdown(
        """
        **Project:** Clinical Trial Disease Category Classification

        **Technology:** Python, Scikit-learn, TF-IDF, Streamlit

        **Final Model:** Logistic Regression

        **Previously evaluated test accuracy:** 95.00%

        The model predicts a disease category based on the
        text provided. Predictions should be reviewed and
        are not a substitute for clinical assessment.
        """
    )