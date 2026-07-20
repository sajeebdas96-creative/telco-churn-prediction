# -*- coding: utf-8 -*-

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# =========================================================
# PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="Telco Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)


# =========================================================
# LOAD TRAINED MODEL
# =========================================================
MODEL_PATH = (
    Path(__file__).resolve().parent
    / "telco_churn_model.pkl"
)


@st.cache_resource
def load_model():
    """Load the saved preprocessing and classification pipeline."""
    return joblib.load(MODEL_PATH)


try:
    model = load_model()

except FileNotFoundError:
    st.error(
        "The model file 'telco_churn_model.pkl' was not found. "
        "Confirm that it is in the same repository folder as app.py."
    )
    st.stop()

except Exception as error:
    st.error("The trained model could not be loaded.")
    st.exception(error)
    st.stop()


# =========================================================
# APPLICATION HEADING
# =========================================================
st.title("Telco Customer Churn Prediction")

st.write(
    """
    This application estimates the probability that a telecommunications
    customer will discontinue their service. Enter the customer information
    in the sidebar and select **Predict Churn Risk**.
    """
)

st.info(
    """
    The result is a probabilistic estimate. It does not confirm that an
    individual customer will or will not churn.
    """
)


# =========================================================
# CUSTOMER INPUTS
# =========================================================
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
    "Partner",
    ["No", "Yes"]
)

dependents = st.sidebar.selectbox(
    "Dependents",
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
        "Multiple Lines has automatically been set to "
        "'No phone service'."
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
        "Internet-related services have automatically been set to "
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
    min_value=18.25,
    max_value=118.75,
    value=70.00,
    step=0.05,
    format="%.2f"
)

if tenure == 0:
    total_charges = 0.0

    st.sidebar.info(
        "Total Charges has been set to 0 because tenure is 0 months."
    )

else:
    total_charges = st.sidebar.number_input(
        "Total Charges",
        min_value=0.0,
        max_value=8684.80,
        value=800.00,
        step=10.00,
        format="%.2f"
    )


# =========================================================
# CREATE MODEL INPUT
# Column names match the training dataset exactly.
# =========================================================
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


# =========================================================
# DISPLAY CUSTOMER PROFILE
# =========================================================
st.subheader("Customer Profile")

profile_table = pd.DataFrame({
    "Customer Characteristic": [
        "Gender",
        "Senior Citizen",
        "Partner",
        "Dependents",
        "Tenure",
        "Phone Service",
        "Multiple Lines",
        "Internet Service",
        "Online Security",
        "Online Backup",
        "Device Protection",
        "Technical Support",
        "Streaming TV",
        "Streaming Movies",
        "Contract",
        "Paperless Billing",
        "Payment Method",
        "Monthly Charges",
        "Total Charges"
    ],
    "Selected Value": [
        gender,
        senior_citizen,
        partner,
        dependents,
        f"{tenure} months",
        phone_service,
        multiple_lines,
        internet_service,
        online_security,
        online_backup,
        device_protection,
        tech_support,
        streaming_tv,
        streaming_movies,
        contract,
        paperless_billing,
        payment_method,
        f"{monthly_charges:.2f}",
        f"{total_charges:.2f}"
    ]
})

st.dataframe(
    profile_table,
    hide_index=True,
    use_container_width=True
)


# =========================================================
# GENERATE PREDICTION
# =========================================================
if st.button(
    "Predict Churn Risk",
    type="primary",
    use_container_width=True
):

    try:
        churn_probability = float(
            model.predict_proba(customer_data)[0, 1]
        )

        decision_threshold = 0.50
        predicted_churn = churn_probability >= decision_threshold

        # Descriptive risk categories
        if churn_probability < 0.35:
            risk_category = "Low"

        elif churn_probability < 0.65:
            risk_category = "Moderate"

        else:
            risk_category = "High"

        if predicted_churn:
            model_classification = "Elevated Churn Risk"
        else:
            model_classification = "Lower Churn Risk"


        st.subheader("Prediction Result")

        column_1, column_2, column_3 = st.columns(3)

        with column_1:
            st.metric(
                "Estimated Churn Probability",
                f"{churn_probability * 100:.1f}%"
            )

        with column_2:
            st.metric(
                "Risk Category",
                risk_category
            )

        with column_3:
            st.metric(
                "Model Classification",
                model_classification
            )


        st.progress(
            churn_probability,
            text=(
                f"Estimated churn probability: "
                f"{churn_probability * 100:.1f}%"
            )
        )


        if predicted_churn:
            st.warning(
                """
                The probability is at or above the model's 50% decision
                threshold. The customer has therefore been classified as
                having elevated churn risk. This does not confirm that the
                customer will churn.
                """
            )

        else:
            st.success(
                """
                The probability is below the model's 50% decision threshold.
                The customer has therefore been classified as having lower
                churn risk. This does not guarantee that the customer will
                remain.
                """
            )


        with st.expander("How to interpret this result"):

            st.markdown(
                """
                **Model classification**

                - Below 50%: lower predicted churn risk
                - 50% or above: elevated predicted churn risk

                **Risk categories**

                - Below 35%: Low
                - 35% to below 65%: Moderate
                - 65% or above: High

                The risk category describes the probability level. The binary
                classification uses the model's 50% decision threshold.
                """
            )

            st.write(
                """
                A customer may have an actual historical outcome of No Churn
                even when the model predicts elevated risk. This is a false
                positive. The model may also miss a customer who actually
                churns, which is a false negative.
                """
            )


    except Exception as error:
        st.error("The prediction could not be generated.")
        st.exception(error)


# =========================================================
# MODEL PERFORMANCE
# =========================================================
st.divider()

with st.expander("Validated Model Performance"):

    performance_data = pd.DataFrame({
        "Metric": [
            "Accuracy",
            "Precision",
            "Recall",
            "F1-score",
            "ROC-AUC"
        ],
        "Test Result": [
            "80.55%",
            "65.72%",
            "55.88%",
            "60.40%",
            "84.20%"
        ]
    })

    st.dataframe(
        performance_data,
        hide_index=True,
        use_container_width=True
    )

    st.write(
        """
        The model was evaluated on 1,409 test customers. It correctly
        identified 209 customers who churned, missed 165 customers who
        churned and incorrectly flagged 109 customers who did not churn.
        """
    )


# =========================================================
# RESPONSIBLE-USE NOTICE
# =========================================================
st.caption(
    """
    Academic demonstration only. Predictions should support customer
    retention analysis and should not replace professional judgement or be
    used as the sole basis for customer treatment, service restrictions or
    financial decisions.
    """
)
