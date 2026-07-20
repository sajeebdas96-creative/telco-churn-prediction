from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Telco Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)


# ---------------------------------------------------------
# Load trained model
# ---------------------------------------------------------
MODEL_PATH = (
    Path(__file__).resolve().parent
    / "telco_churn_model.pkl"
)


@st.cache_resource
def load_model():
    """Load the trained preprocessing and classification pipeline."""
    return joblib.load(MODEL_PATH)


try:
    model = load_model()

except Exception as error:
    st.error("The trained model could not be loaded.")
    st.exception(error)
    st.stop()


# ---------------------------------------------------------
# Application heading
# ---------------------------------------------------------
st.title("Telco Customer Churn Prediction")

st.write(
    """
    This application estimates whether a telecommunications customer
    is likely to discontinue their service. Enter the customer profile
    in the sidebar and select **Predict Churn Risk**.
    """
)


# ---------------------------------------------------------
# Sidebar inputs
# ---------------------------------------------------------
st.sidebar.header("Customer Information")

gender = st.sidebar.selectbox(
    "Gender",
    ["Female", "Male"]
)

senior_citizen = st.sidebar.selectbox(
    "Senior Citizen",
    ["No", "Yes"]
)

partner = st.sidebar.selectbox(
    "Has a Partner",
    ["No", "Yes"]
)

dependents = st.sidebar.selectbox(
    "Has Dependents",
    ["No", "Yes"]
)

tenure = st.sidebar.slider(
    "Tenure in Months",
    min_value=0,
    max_value=72,
    value=12
)

phone_service = st.sidebar.selectbox(
    "Phone Service",
    ["Yes", "No"]
)

if phone_service == "No":
    multiple_lines = "No phone service"
    st.sidebar.info(
        "Multiple Lines automatically set to 'No phone service'."
    )
else:
    multiple_lines = st.sidebar.selectbox(
        "Multiple Lines",
        ["No", "Yes"]
    )

internet_service = st.sidebar.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"]
)

if internet_service == "No":
    online_security = "No internet service"
    online_backup = "No internet service"
    device_protection = "No internet service"
    tech_support = "No internet service"
    streaming_tv = "No internet service"
    streaming_movies = "No internet service"

    st.sidebar.info(
        "Internet-related services automatically set to "
        "'No internet service'."
    )

else:
    online_security = st.sidebar.selectbox(
        "Online Security",
        ["No", "Yes"]
    )

    online_backup = st.sidebar.selectbox(
        "Online Backup",
        ["No", "Yes"]
    )

    device_protection = st.sidebar.selectbox(
        "Device Protection",
        ["No", "Yes"]
    )

    tech_support = st.sidebar.selectbox(
        "Technical Support",
        ["No", "Yes"]
    )

    streaming_tv = st.sidebar.selectbox(
        "Streaming TV",
        ["No", "Yes"]
    )

    streaming_movies = st.sidebar.selectbox(
        "Streaming Movies",
        ["No", "Yes"]
    )

contract = st.sidebar.selectbox(
    "Contract Type",
    [
        "Month-to-month",
        "One year",
        "Two year"
    ]
)

paperless_billing = st.sidebar.selectbox(
    "Paperless Billing",
    ["Yes", "No"]
)

payment_method = st.sidebar.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)

monthly_charges = st.sidebar.number_input(
    "Monthly Charges",
    min_value=0.0,
    max_value=150.0,
    value=70.0,
    step=1.0
)

total_charges = st.sidebar.number_input(
    "Total Charges",
    min_value=0.0,
    max_value=10000.0,
    value=800.0,
    step=10.0
)


# ---------------------------------------------------------
# Prepare input data
# ---------------------------------------------------------
customer_data = pd.DataFrame({
    "gender": [gender],
    "SeniorCitizen": [senior_citizen],
    "Partner": [partner],
    "Dependents": [dependents],
    "tenure": [tenure],
    "PhoneService": [phone_service],
    "MultipleLines": [multiple_lines],
    "InternetService": [internet_service],
    "OnlineSecurity": [online_security],
    "OnlineBackup": [online_backup],
    "DeviceProtection": [device_protection],
    "TechSupport": [tech_support],
    "StreamingTV": [streaming_tv],
    "StreamingMovies": [streaming_movies],
    "Contract": [contract],
    "PaperlessBilling": [paperless_billing],
    "PaymentMethod": [payment_method],
    "MonthlyCharges": [monthly_charges],
    "TotalCharges": [total_charges]
})


# ---------------------------------------------------------
# Display customer profile
# ---------------------------------------------------------
st.subheader("Customer Profile")

st.dataframe(
    customer_data,
    use_container_width=True,
    hide_index=True
)


# ---------------------------------------------------------
# Generate prediction
# ---------------------------------------------------------
if st.button(
    "Predict Churn Risk",
    type="primary",
    use_container_width=True
):
    try:
        prediction = int(
            model.predict(customer_data)[0]
        )

        churn_probability = float(
            model.predict_proba(customer_data)[0, 1]
        )

        if churn_probability < 0.30:
            risk_level = "Low"

        elif churn_probability < 0.60:
            risk_level = "Moderate"

        else:
            risk_level = "High"

        st.subheader("Prediction Result")

        metric_column_1, metric_column_2 = st.columns(2)

        with metric_column_1:
            st.metric(
                "Estimated Churn Probability",
                f"{churn_probability * 100:.1f}%"
            )

        with metric_column_2:
            st.metric(
                "Risk Category",
                risk_level
            )

        st.progress(churn_probability)

        if prediction == 1:
            st.warning(
                "The model predicts that this customer is "
                "at risk of churn."
            )

        else:
            st.success(
                "The model predicts that this customer is "
                "likely to remain."
            )

    except Exception as error:
        st.error("The prediction could not be generated.")
        st.exception(error)


# ---------------------------------------------------------
# Responsible-use notice
# ---------------------------------------------------------
st.divider()

st.caption(
    """
    Academic demonstration only. The prediction should support
    customer-retention decisions and should not replace professional
    judgement or be used as the sole basis for customer treatment.
    """
)
