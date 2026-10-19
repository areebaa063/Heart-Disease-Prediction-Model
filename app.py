
"""Heart Disease Prediction - Streamlit web app

Loads the trained Support Vector Classifier saved by the notebook
(heart_disease_svc_model.pkl) and lets users try it in the browser.
"""

import pickle
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder

# ---- Page Configuration -------------------------------------------------------
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="centered",
)

# ---- Author Credit ------------------------------------------------------------
GITHUB_URL = "https://github.com/areebaa063"

# Put your notebook's test accuracy here (for example 0.78). Leave None to hide it.
TEST_ACCURACY = None

MODEL_PATH = Path(__file__).parent / "heart_disease_svc_model.pkl"

# ---- Category Lists (must match the notebook) ---------------------------------
SEX_OPTIONS = ["Male", "Female"]

CHEST_PAIN_OPTIONS = [
    "Typical angina",
    "Atypical angina",
    "Non-anginal pain",
    "Asymptomatic",
]

EXERCISE_ANGINA_OPTIONS = ["Yes", "No"]

HEART_RATE_OPTIONS = ["Low", "Medium", "High"]

HEART_RATE_LABELS = {
    "Low": "Low (below 140 bpm)",
    "Medium": "Medium (140-159 bpm)",
    "High": "High (160 bpm or more)",
}


# ---- Load Model ---------------------------------------------------------------
@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


@st.cache_resource
def build_label_encoders():
    """Re-create the same LabelEncoders as the notebook."""
    return {
        "Sex": LabelEncoder().fit(SEX_OPTIONS),
        "ChestPain": LabelEncoder().fit(CHEST_PAIN_OPTIONS),
        "ExerciseAngina": LabelEncoder().fit(EXERCISE_ANGINA_OPTIONS),
        "MaxHeartRate": LabelEncoder().fit(HEART_RATE_OPTIONS),
    }


model = load_model()
encoders = build_label_encoders()

# ---- Header -------------------------------------------------------------------
header_col1, header_col2 = st.columns([4, 1])

with header_col1:
    st.title("❤️ Heart Disease Predictor")

with header_col2:
    st.markdown(
        f"""
        <div style="display: flex; justify-content: flex-end; padding-top: 20px;">
            <a href="{GITHUB_URL}" target="_blank"
               title="Visit Areeba Butt's GitHub"
               style="text-decoration: none; color: inherit;">
                <svg xmlns="http://www.w3.org/2000/svg"
                     width="34" height="34" viewBox="0 0 24 24"
                     fill="currentColor" aria-label="GitHub">
                    <path d="M12 .9a11.1 11.1 0 0 0-3.51 21.63c.55.1.76-.24.76-.54
                    v-2.1c-3.1.67-3.76-1.31-3.76-1.31-.5-1.29-1.24-1.63-1.24-1.63
                    -1.02-.7.08-.69.08-.69 1.13.08 1.72 1.16 1.72 1.16 1 .1.77 2.01
                    2.95 2.95.1-.73.39-1.23.7-1.52-2.48-.28-5.09-1.24-5.09-5.53
                    0-1.22.44-2.22 1.16-3-.12-.28-.5-1.42.11-2.96 0 0 .95-.3 3.05
                    1.15a10.6 10.6 0 0 1 5.55 0c2.1-1.45 3.05-1.15 3.05-1.15
                    .61 1.54.23 2.68.11 2.96.72.78 1.16 1.78 1.16 3
                    0 4.3-2.62 5.24-5.11 5.52.4.35.76 1.03.76 2.08v3.08
                    c0 .3.2.65.77.54A11.1 11.1 0 0 0 12 .9Z"/>
                </svg>
            </a>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    f"""
    <div style="text-align: right; margin-top: -10px; margin-bottom: 20px;
                font-size: 14px; color: #808495;">
        Created by
        <a href="{GITHUB_URL}" target="_blank"
           style="color: inherit; font-weight: 600; text-decoration: none;">
           Areeba Butt
        </a>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---- Introduction -------------------------------------------------------------
st.markdown(
    "Answer the 4 questions below and the machine learning model will estimate "
    "whether heart disease is likely. Built on the UCI Heart Disease (Cleveland) dataset."
)

if TEST_ACCURACY is not None:
    st.markdown(
        f"*Model accuracy on held-out test data: **{TEST_ACCURACY:.0%}***"
    )

# ---- Input Form ---------------------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    sex = st.selectbox("Sex", SEX_OPTIONS)
    chest_pain = st.selectbox(
        "Chest Pain Type",
        CHEST_PAIN_OPTIONS,
        index=3,
    )

with col2:
    exercise_angina = st.selectbox(
        "Exercise-Induced Angina",
        EXERCISE_ANGINA_OPTIONS,
        index=1,
    )

    max_heart_rate = st.selectbox(
        "Max Heart Rate Achieved",
        HEART_RATE_OPTIONS,
        index=1,
        format_func=lambda option: HEART_RATE_LABELS[option],
    )

# ---- Prediction ---------------------------------------------------------------
if st.button("Predict", type="primary", use_container_width=True):

    # Column names and order must match the training data
    user_input = pd.DataFrame({
        "Sex": [sex],
        "ChestPain": [chest_pain],
        "ExerciseAngina": [exercise_angina],
        "MaxHeartRate": [max_heart_rate],
    })

    # Encode inputs using the same categories as the notebook
    encoded = pd.DataFrame({
        "Sex": encoders["Sex"].transform(user_input["Sex"]),
        "ChestPain": encoders["ChestPain"].transform(user_input["ChestPain"]),
        "ExerciseAngina": encoders["ExerciseAngina"].transform(
            user_input["ExerciseAngina"]
        ),
        "MaxHeartRate": encoders["MaxHeartRate"].transform(
            user_input["MaxHeartRate"]
        ),
    })

    prediction = model.predict(encoded)[0]

    st.divider()

    if prediction == 1:
        st.error("### ⚠️ HEART DISEASE LIKELY")
    else:
        st.success("### ✅ HEART DISEASE NOT LIKELY")

    with st.expander("See the encoded feature vector sent to the model"):
        st.dataframe(encoded, hide_index=True)

# ---- Footer -------------------------------------------------------------------
st.caption(
    "Educational project for a Machine Learning course. Uses only 4 inputs and "
    "a small dataset. This is NOT medical advice — see a doctor for any health concern."
)
