import sys
from pathlib import Path

import joblib
import streamlit as st


st.set_page_config(
    page_title="Telco Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# This must appear before the model is loaded
st.title("Telco Customer Churn Prediction")
st.success("The Streamlit interface loaded successfully.")

st.write("Python version:", sys.version)

# Locate the model relative to app.py
MODEL_PATH = (
    Path(__file__).resolve().parent
    / "telco_churn_model.pkl"
)

st.write("Model file path:", str(MODEL_PATH))
st.write("Model file exists:", MODEL_PATH.exists())


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


try:
    model = load_model()
    st.success("The trained model loaded successfully.")

except Exception as error:
    st.error("The application loaded, but the model could not be opened.")
    st.exception(error)
    st.stop()


st.info(
    "Diagnostic test completed. The application and model are working."
)
