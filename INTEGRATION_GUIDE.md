# Streamlit App Integration with MLflow Model & SHAP

## Overview

The Streamlit application (`app/App.py`) has been integrated with the loan default prediction model from `predict.py`. The integration includes:

1. **Real Model Predictions**: Load the trained LightGBM model from MLflow
2. **SHAP-based Explainability**: Compute SHAP values to explain model predictions
3. **Business Threshold Logic**: Apply configurable business thresholds for loan approval/rejection
4. **Interactive Frontend**: User-friendly Streamlit interface for making predictions

## Architecture

### Key Components

#### 1. **model_utils.py** (New Utility Module)

Located in `src/model_utils.py`, this module contains reusable functions extracted from `predict.py`:

- `load_model()` - Load trained model from MLflow
- `load_background_data()` - Load background data for SHAP explainer
- `initialize_shap_explainer()` - Initialize SHAP TreeExplainer
- `compute_shap_values()` - Compute SHAP values for feature importance
- `predict()` - Make predictions on new data
- `validate_input()` - Validate input data structure
- `evaluate_predictions()` - Calculate performance metrics

#### 2. **app/App.py** (Updated Streamlit App)

The main application now includes:

- **Model Caching**: Uses `@st.cache_resource` to cache model, explainer, and background data
- **Real Predictions**: Converts user input to DataFrame and calls actual model
- **SHAP Integration**: Computes and visualizes SHAP values for each prediction
- **Three Tabs**:
  - **Input Details**: Collect applicant information
  - **Prediction Results**: Display prediction probability and risk gauge
  - **Risk Analysis**: Show SHAP-based feature importance and business decision

#### 3. **src/predict.py** (Updated)

Now imports functions from `model_utils.py` for cleaner code organization.

## Integration Flow

```
User Input (App.py)
    ↓
Create DataFrame (matching model schema)
    ↓
Validate Input (model_utils.validate_input)
    ↓
Get Predictions (model_utils.predict)
    ↓
Compute SHAP Values (model_utils.compute_shap_values)
    ↓
Display Results & Risk Analysis
    ↓
Apply Business Threshold (configurable in UI)
    ↓
Final Verdict (APPROVE/REJECT)
```

## File Structure

```
d:/projects/home_default/
├── app/
│   └── App.py                 # Main Streamlit application (updated)
├── src/
│   ├── model_utils.py         # Reusable utility functions (NEW)
│   ├── predict.py             # Original prediction script (updated)
│   └── train.py
├── Dataset/
│   ├── Loan_default.csv       # Full dataset (for SHAP background)
│   ├── x_test.csv
│   └── y_test.csv
├── mlruns/                     # MLflow tracking directory
├── requirements.txt
└── INTEGRATION_GUIDE.md        # This file
```

## Required Dependencies

The following packages are required:

```
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
lightgbm>=4.0.0
mlflow>=2.0.0
shap>=0.50.0
matplotlib>=3.7.0
seaborn>=0.12.0
```

Install with:

```bash
pip install -r requirements.txt
```

## How to Run

### 1. Run the Streamlit App

```bash
cd d:/projects/home_default
streamlit run app/App.py
```

The app will start at `http://localhost:8501`

### 2. Using the App

1. **Home Page**: Overview and model information
2. **Documentation Page**: Feature descriptions and SHAP explanation
3. **Prediction Page**:
   - Fill in applicant details
   - Click "Get Prediction" to see:
     - Default probability
     - Risk level (Low/Medium/High)
     - SHAP-based feature importance
     - Business threshold logic
     - Final verdict

### 3. Running the Original Prediction Script

```bash
cd d:/projects/home_default
python src/predict.py
```

This will:

- Load test data
- Make predictions
- Compute SHAP values
- Save results to `output/` directory

## Key Features

### Real-time Model Loading

- Model is cached on first app load
- SHAP explainer initialized once and reused
- Background data loaded and cached for efficiency

### SHAP Explainability

- Shows top 8 features by importance
- Indicates direction of impact (positive/negative)
- Displays SHAP values for each feature
- Top 3 risk factors highlighted

### Business Decision Logic

- **Configurable Threshold**: Adjust in the UI (default 0.45)
- **Approval/Rejection**: Automatic based on probability vs threshold
- **Safety Margin**: Shows distance from threshold
- **Risk Assessment**: Categorized as Low/Medium/High

### Input Validation

- Checks for missing columns
- Validates data types
- Warns on null values
- Ensures data consistency with training data

## Integration Details

### Data Flow for Predictions

```python
# User enters these features in the app:
input_data = {
    "Age": 35,
    "Income": 50000,
    "LoanAmount": 25000,
    "CreditScore": 650,
    "MonthsEmployed": 60,
    "NumCreditLines": 3,
    "LoanTerm": 60,
    "Education": "Bachelor's",
    "EmploymentType": "Full-time",
    "MaritalStatus": "Married",
    "HasMortgage": 1,
    "HasDependents": 1,
    "LoanPurpose": "Home",
    "HasCoSigner": 0,
}

# Converted to DataFrame
X_test = pd.DataFrame([input_data])

# Predictions
y_pred, y_pred_proba = predict(model, X_test)
# y_pred: [0 or 1] (binary prediction)
# y_pred_proba: [0.0-1.0] (probability of default)

# SHAP explanations
shap_values, feature_importance_df = compute_shap_values(
    explainer, model, X_test, X_background_transformed
)
```
 
### Caching Strategy

Three resource caching functions ensure efficiency:

```python
@st.cache_resource
def load_cached_model():
    model, run_id = load_model()
    return model, run_id

@st.cache_resource
def load_cached_background_data():
    X_background = load_background_data(sample_size=100)
    return X_background

@st.cache_resource
def load_cached_explainer(model, X_background):
    explainer, X_background_transformed = initialize_shap_explainer(...)
    return explainer, X_background_transformed
```

## Troubleshooting

### Issue: "Model failed to load"

- **Solution**: Check MLflow tracking directory path and experiment name
- Verify `mlruns/` directory exists with the correct structure
- Check that "loan_default_experiment" and "Pipeline_LightGBM_port" exist in MLflow

### Issue: "Error loading background data"

- **Solution**: Ensure `Dataset/Loan_default.csv` exists
- Check file path is correct relative to working directory
- Verify CSV format and columns match training data

### Issue: SHAP computation errors

- **Solution**: May fall back to KernelExplainer (slower but more robust)
- Ensure background data is properly formatted
- Check that model and X_test have matching feature columns

### Issue: Import errors

- **Solution**: Install all dependencies from requirements.txt
- Ensure `src/` directory is in Python path
- Check that model_utils.py is in `src/` directory

## Future Enhancements

1. **Batch Predictions**: Upload CSV file for multiple predictions
2. **Model Comparison**: Compare multiple model versions from MLflow
3. **Performance Analytics**: Historical predictions and accuracy metrics
4. **Custom Thresholds**: Save and manage multiple business thresholds
5. **Export Reports**: Generate PDF reports for each prediction
6. **A/B Testing**: Compare different model versions on live data

## Performance Notes

- **First Load**: Slower due to model and explainer initialization (30-60 seconds)
- **Subsequent Loads**: Fast due to caching (1-2 seconds)
- **Prediction Time**: 1-3 seconds per prediction (including SHAP computation)
- **Background Data**: 100 samples recommended for balance between accuracy and speed

## References

- [Streamlit Documentation](https://docs.streamlit.io/)
- [SHAP Documentation](https://shap.readthedocs.io/)
- [MLflow Documentation](https://mlflow.org/docs/)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
