import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import sys
import os
from pathlib import Path

# Add src directory to path to import model_utils
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from model_utils import (
    load_model,
    load_background_data,
    initialize_shap_explainer,
    compute_shap_values,
    predict,
    validate_input,
)

# Set page config
st.set_page_config(
    page_title="Loan Default Prediction",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for styling
st.markdown(
    """
    <style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .prediction-pass {
        background-color: #d4edda;
        border: 2px solid #28a745;
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
    }
    .prediction-fail {
        background-color: #f8d7da;
        border: 2px solid #dc3545;
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
    }
    </style>
""",
    unsafe_allow_html=True,
)


# Cache model loading for efficiency
@st.cache_resource
def load_cached_model():
    """Load and cache the model from MLflow"""
    try:
        model, run_id = load_model()
        return model, run_id
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None


@st.cache_resource
def load_cached_background_data():
    """Load and cache background data for SHAP"""
    try:
        X_background = load_background_data(sample_size=100)
        return X_background
    except Exception as e:
        st.error(f"Error loading background data: {e}")
        return None


@st.cache_resource
def load_cached_explainer(_model, X_background):
    """Load and cache SHAP explainer"""
    if _model is None or X_background is None:
        return None, None
    try:
        explainer, X_background_transformed = initialize_shap_explainer(
            _model, X_background
        )
        return explainer, X_background_transformed
    except Exception as e:
        st.error(f"Error initializing explainer: {e}")
        return None, None


# Load model and explainer once
with st.spinner("Loading model and SHAP explainer..."):
    model, run_id = load_cached_model()
    X_background = load_cached_background_data()
    explainer, X_background_transformed = load_cached_explainer(model, X_background)

# App Title
st.title("🏦 Loan Default Risk Assessment")
st.markdown("---")

# Sidebar for navigation
with st.sidebar:
    st.header("Navigation")
    page = st.radio("Select a page:", ["Home", "Prediction", "Documentation"])

# Home Page
if page == "Home":
    st.header("Welcome to Loan Default Risk Assessment Tool")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Model Status", "Active", "✓ Ready")
    with col2:
        st.metric("Business Threshold", "0.85", "Risk Level")
    with col3:
        st.metric("Last Updated", datetime.now().strftime("%Y-%m-%d"), "Today")

    st.markdown(
        """
    ### About This Application
    
    This application helps assess the risk of loan default using machine learning.
    It provides:
    
    - **Prediction Results**: Probability of loan default
    - **Feature Importance**: Key factors affecting the decision using SHAP values
    - **Business Verdict**: Pass/Reject recommendation based on threshold
    
    ### How to Use
    
    1. Navigate to the **Prediction** page
    2. Enter applicant details
    3. Review the prediction and SHAP-based feature importance
    4. Make an informed decision
    
    ### Model Information
    - **Run ID**: {0}
    - **Model Type**: LightGBM Pipeline
    - **Explainability**: SHAP (SHapley Additive exPlanations)
    """.format(
            run_id if run_id else "Loading..."
        )
    )

# Documentation Page
elif page == "Documentation":
    st.header("📚 Documentation")

    st.subheader("Input Features")

    features_info = {
        "Age": "Applicant's age in years (18-80)",
        "Income": "Annual income in USD",
        "LoanAmount": "Requested loan amount in USD",
        "CreditScore": "Credit score (300-850)",
        "MonthsEmployed": "Months at current employment (0+)",
        "NumCreditLines": "Number of active credit lines (0+)",
        "LoanTerm": "Loan term in months",
        "Education": "Educational level",
        "EmploymentType": "Type of employment",
        "MaritalStatus": "Marital status",
        "HasMortgage": "Whether applicant has mortgage",
        "HasDependents": "Whether applicant has dependents",
        "LoanPurpose": "Purpose of loan",
        "HasCoSigner": "Whether loan has co-signer",
    }

    for feature, description in features_info.items():
        st.write(f"**{feature}**: {description}")

    st.markdown(
        """
    ### About SHAP Values
    
    SHAP (SHapley Additive exPlanations) values provide a unified measure of feature importance.
    They show how much each feature contributes to pushing the prediction from the base value
    (average model output) to the actual prediction.
    
    - **Positive SHAP values**: Push the prediction towards higher default probability
    - **Negative SHAP values**: Push the prediction towards lower default probability
    - **Larger absolute values**: More important features
    """
    )

# Prediction Page
elif page == "Prediction":
    if model is None:
        st.error("❌ Model failed to load. Please check the MLflow configuration.")
    else:
        st.header("🔮 Make a Prediction")

        # Create tabs for input and results
        tab1, tab2, tab3 = st.tabs(
            ["Input Details", "Prediction Results", "Risk Analysis"]
        )

        # TAB 1: Input Details
        with tab1:
            st.subheader("Enter Applicant Details")

            col1, col2, col3 = st.columns(3)

            with col1:
                age = st.number_input("Age", min_value=18, max_value=80, value=35)
                income = st.number_input(
                    "Annual Income (USD)", min_value=0, value=50000, step=1000
                )
                loan_amount = st.number_input(
                    "Loan Amount (USD)", min_value=0, value=25000, step=1000
                )
                credit_score = st.number_input(
                    "Credit Score", min_value=300, max_value=850, value=650
                )
                months_employed = st.number_input(
                    "Months Employed", min_value=0, max_value=600, value=60
                )

            with col2:
                num_credit_lines = st.number_input(
                    "Number of Credit Lines", min_value=0, max_value=20, value=3
                )
                loan_term = st.number_input(
                    "Loan Term (months)", min_value=6, max_value=360, value=60, step=6
                )
                education = st.selectbox(
                    "Education Level",
                    ["High School", "Associate's", "Bachelor's", "Master's", "PhD"],
                )
                employment_type = st.selectbox(
                    "Employment Type",
                    [
                        "Full-time",
                        "Part-time",
                        "Self-employed",
                        "Unemployed",
                        "Retired",
                    ],
                )
                marital_status = st.selectbox(
                    "Marital Status", ["Single", "Married", "Divorced", "Widowed"]
                )

            with col3:
                has_mortgage = st.selectbox("Has Mortgage?", ["No", "Yes"])
                has_dependents = st.selectbox("Has Dependents?", ["No", "Yes"])
                loan_purpose = st.selectbox(
                    "Loan Purpose",
                    [
                        "Home",
                        "Auto",
                        "Business",
                        "Debt Consolidation",
                        "Education",
                        "Other",
                    ],
                )
                has_cosigner = st.selectbox("Has Co-Signer?", ["No", "Yes"])

            # Submit button
            col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 2])
            with col_btn1:
                predict_button = st.button(
                    "🚀 Get Prediction", use_container_width=True
                )
            with col_btn2:
                clear_button = st.button("🔄 Clear Form", use_container_width=True)

            if clear_button:
                st.rerun()

        # Prepare input data
        input_data = None
        prediction_proba = None
        shap_values = None
        feature_importance_df = None

        if predict_button:
            try:
                # Create DataFrame from user input
                input_data = pd.DataFrame(
                    {
                        "Age": [int(age)],
                        "Income": [float(income)],
                        "LoanAmount": [float(loan_amount)],
                        "CreditScore": [int(credit_score)],
                        "MonthsEmployed": [int(months_employed)],
                        "NumCreditLines": [int(num_credit_lines)],
                        "LoanTerm": [int(loan_term)],
                        "Education": [education],
                        "EmploymentType": [employment_type],
                        "MaritalStatus": [marital_status],
                        "HasMortgage": [has_mortgage],
                        "HasDependents": [has_dependents],
                        "LoanPurpose": [loan_purpose],
                        "HasCoSigner": [has_cosigner],
                    }
                )

                # Validate input
                validate_input(input_data)

                # Get predictions
                with st.spinner("Making predictions..."):
                    y_pred, y_pred_proba = predict(model, input_data)
                    prediction_proba = y_pred_proba[0]

                # Compute SHAP values
                if explainer is not None:
                    with st.spinner("Computing SHAP values for explanations..."):
                        shap_values, feature_importance_df = compute_shap_values(
                            explainer, model, input_data, X_background_transformed
                        )
                        shap_values = shap_values[
                            0
                        ]  # Get values for first (only) sample

            except Exception as e:
                st.error(f"Error during prediction: {e}")
                st.stop()

        # TAB 2: Prediction Results
        with tab2:
            if predict_button and input_data is not None:
                st.subheader("Prediction Results")

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(
                        "Default Probability",
                        f"{prediction_proba:.2%}",
                        f"{prediction_proba - 0.85:.2%}",
                    )

                with col2:
                    risk_level = (
                        "High"
                        if prediction_proba > 0.6
                        else "Medium" if prediction_proba > 0.4 else "Low"
                    )
                    risk_emoji = (
                        "🔴"
                        if prediction_proba > 0.6
                        else "🟡" if prediction_proba > 0.4 else "🟢"
                    )
                    st.metric("Risk Level", risk_level, risk_emoji)

                with col3:
                    confidence = (1 - abs(prediction_proba - 0.5) * 2) * 100
                    st.metric(
                        "Model Confidence",
                        f"{confidence:.1f}%",
                        "High" if confidence > 70 else "Medium",
                    )

                st.markdown("---")

                # Prediction gauge
                st.subheader("Default Risk Gauge")

                fig, ax = plt.subplots(figsize=(10, 6))

                # Create gauge chart
                categories = [
                    "Very Low\n(0-0.2)",
                    "Low\n(0.2-0.4)",
                    "Medium\n(0.4-0.6)",
                    "High\n(0.6-0.8)",
                    "Very High\n(0.8-1.0)",
                ]

                colors = ["#28a745", "#5cb85c", "#ffc107", "#fd7e14", "#dc3545"]

                ax.barh(categories, [1, 1, 1, 1, 1], color=colors, alpha=0.7)

                # Add pointer
                risk_category = int(prediction_proba * 5)
                ax.scatter(
                    [prediction_proba],
                    [risk_category],
                    color="black",
                    s=500,
                    marker=">",
                    zorder=10,
                )

                ax.set_xlim(0, 1)
                ax.set_xlabel("Probability of Default", fontsize=12)
                ax.set_title("Risk Assessment Gauge", fontsize=14, fontweight="bold")
                ax.grid(axis="x", alpha=0.3)

                st.pyplot(fig, use_container_width=True)

            else:
                st.info(
                    "👈 Please fill in the details and click 'Get Prediction' to see results"
                )

        # TAB 3: Risk Analysis (SHAP + Business Threshold)
        with tab3:
            if (
                predict_button
                and input_data is not None
                and feature_importance_df is not None
            ):
                st.subheader("Feature Importance & Contributing Factors (SHAP)")

                # Create SHAP-like visualization
                fig, ax = plt.subplots(figsize=(10, 6))

                # Get top features and their SHAP values for this sample
                feature_names = feature_importance_df["feature"].tolist()[:8]
                shap_sample_values = shap_values[: len(feature_names)]

                # Find indices of top features in original feature list
                sorted_indices = np.argsort(np.abs(shap_sample_values))[::-1][:8]
                top_features = [feature_names[i] for i in sorted_indices]
                top_shap_values = [shap_sample_values[i] for i in sorted_indices]

                colors = ["#d62728" if v > 0 else "#1f77b4" for v in top_shap_values]
                ax.barh(top_features, top_shap_values, color=colors, alpha=0.7)

                ax.set_xlabel("SHAP Value (Impact on Default Probability)", fontsize=11)
                ax.set_title(
                    "Feature Importance - Contributing Factors",
                    fontsize=13,
                    fontweight="bold",
                )
                ax.axvline(x=0, color="black", linestyle="-", linewidth=0.8)
                ax.grid(axis="x", alpha=0.3)

                # Add value labels
                for i, v in enumerate(top_shap_values):
                    ax.text(v, i, f" {v:.3f}", va="center", fontsize=9)

                st.pyplot(fig, use_container_width=True)

                st.markdown("---")

                # Business Threshold Section
                st.subheader("🎯 Business Decision")

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("### Threshold Configuration")
                    business_threshold = st.slider(
                        "Business Threshold",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.85,
                        step=0.05,
                        help="Probability above this threshold results in loan rejection",
                    )

                with col2:
                    st.markdown("### Decision Details")

                    col_metric1, col_metric2 = st.columns(2)
                    with col_metric1:
                        st.metric("Model Probability", f"{prediction_proba:.2%}")
                    with col_metric2:
                        st.metric("Threshold", f"{business_threshold:.2%}")

                st.markdown("---")

                # Final Verdict
                st.subheader("Final Verdict")

                if prediction_proba > business_threshold:
                    st.markdown(
                        f"""
                        <div class="prediction-fail">
                        <h3 style="color: #721c24; margin: 0;">❌ LOAN REJECTED</h3>
                        <p style="color: #721c24; margin: 10px 0 0 0;">
                        The applicant's default probability (<strong>{prediction_proba:.2%}</strong>) 
                        exceeds the business threshold of <strong>{business_threshold:.2%}</strong>
                        </p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    recommendation = "REJECT"
                else:
                    st.markdown(
                        f"""
                        <div class="prediction-pass">
                        <h3 style="color: #155724; margin: 0;">✅ LOAN APPROVED</h3>
                        <p style="color: #155724; margin: 10px 0 0 0;">
                        The applicant's default probability (<strong>{prediction_proba:.2%}</strong>) 
                        is below the business threshold of <strong>{business_threshold:.2%}</strong>
                        </p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    recommendation = "APPROVE"

                # Top Risk Factors
                st.markdown("### Top Risk Factors")

                # Get top 3 features by absolute SHAP value
                sorted_idx = np.argsort(np.abs(shap_values))[::-1][:3]
                risk_factors = []

                for idx in sorted_idx:
                    if idx < len(feature_importance_df):
                        feature = feature_importance_df.iloc[idx]["feature"]
                        shap_value = shap_values[idx] if idx < len(shap_values) else 0
                        direction = "increases" if shap_value > 0 else "decreases"
                        impact = "HIGH" if abs(shap_value) > 0.1 else "MEDIUM"
                        risk_factors.append(
                            f"**{feature}**: {direction} default risk ({impact} impact)"
                        )

                for factor in risk_factors:
                    st.write(f"- {factor}")

                st.markdown("---")

                # Summary Table
                st.subheader("Summary Report")

                summary_data = {
                    "Metric": [
                        "Default Probability",
                        "Business Threshold",
                        "Safety Margin",
                        "Recommendation",
                        "Risk Level",
                    ],
                    "Value": [
                        f"{prediction_proba:.2%}",
                        f"{business_threshold:.2%}",
                        f"{abs(prediction_proba - business_threshold):.2%}",
                        recommendation,
                        (
                            "High"
                            if prediction_proba > 0.6
                            else "Medium" if prediction_proba > 0.4 else "Low"
                        ),
                    ],
                }

                st.dataframe(
                    pd.DataFrame(summary_data),
                    use_container_width=True,
                    hide_index=True,
                )

            else:
                st.info(
                    "👈 Please fill in the details and click 'Get Prediction' to see analysis"
                )

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #888; font-size: 12px;">
    Loan Default Risk Assessment System | Version 2.0 | Integrated with MLflow & SHAP
    </div>
    """,
    unsafe_allow_html=True,
)
