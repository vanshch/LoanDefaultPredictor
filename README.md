# Loan Default Prediction

A machine learning project that predicts loan default risk using LightGBM with SHAP-based explainability. Includes a Streamlit web application for interactive predictions and risk analysis.

## Features

- **ML Model Training**: LightGBM classifier trained on loan applicant data
- **Real-time Predictions**: Make predictions on new loan applications
- **SHAP Explainability**: Understand which features drive each prediction
- **Interactive Web UI**: Streamlit application for exploring predictions
- **Business Logic**: Configurable approval/rejection thresholds
- **Model Tracking**: MLflow integration for experiment tracking and model versioning
- **Comprehensive Metrics**: ROC-AUC, F1-score, precision, recall, and KS statistic

## Quick Start

### Prerequisites

- Python 3.8+
- Git (optional)

### Installation

1. Clone the repository:

   ```bash
   git clone <repository-url>
   cd home_default
   ```

2. Create a virtual environment:

   ```bash
   python -m venv .venv
   source .venv/Scripts/activate  # On Windows
   # or
   source .venv/bin/activate  # On macOS/Linux
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Usage

#### Training a Model

Run the training script to train a new LightGBM model:

```bash
python src/train.py
```

The trained model will be logged to MLflow and saved in the `mlruns/` directory. Training includes:

- Data preprocessing and feature engineering
- Train/test split with stratification
- LightGBM classifier training with hyperparameter tuning
- Performance metrics calculation (ROC-AUC, F1, accuracy, etc.)
- Feature importance tracking

#### Making Predictions

Run the prediction script on test data:

```bash
python src/predict.py
```

This will:

- Load the latest trained model from MLflow
- Load test data from `Dataset/x_test.csv` and `Dataset/y_test.csv`
- Generate predictions with probabilities
- Compute SHAP values for feature importance
- Save results to the `output/` directory

Outputs include:

- `predictions_*.csv` - Predictions with actual values and probabilities
- `metrics_*.csv` - Performance metrics
- `shap_feature_importance_*.csv` - SHAP-based feature importance
- `prediction_summary_*.txt` - Summary report

#### Running the Interactive App

Launch the Streamlit web application:

```bash
streamlit run app/app.py
```

The app includes three tabs:

1. **Input Details**: Enter loan applicant information
2. **Prediction Results**: View prediction probability and risk gauge
3. **Risk Analysis**: Explore SHAP-based feature importance and business decision

The application supports:

- Real-time model predictions
- SHAP value computation for explainability
- Configurable business thresholds for approval decisions
- Interactive visualizations

## Project Structure

```
home_default/
├── README.md                      # This file
├── INTEGRATION_GUIDE.md            # Detailed architecture and integration guide
├── requirements.txt                # Python dependencies
├── app/
│   └── app.py                     # Streamlit web application
├── src/
│   ├── train.py                   # Model training script
│   ├── predict.py                 # Prediction script
│   ├── model_utils.py             # Utility functions for model operations
│   └── artifacts/                 # Generated artifacts (ROC curves, feature importance)
├── Dataset/
│   ├── Loan_default.csv           # Full dataset for training
│   ├── x_test.csv                 # Test features
│   ├── y_test.csv                 # Test labels
│   └── split.py                   # Dataset splitting utility
├── Notebooks/                      # Jupyter notebooks for analysis
│   ├── Model_training.ipynb       # Training pipeline walkthrough
│   ├── EDA-loan-default.ipynb     # Exploratory data analysis
│   ├── explainability.ipynb       # SHAP value analysis
│   ├── Sanity_checks_loan_dataset.ipynb
│   └── Business_cutoff.ipynb      # Business threshold analysis
├── mlruns/                        # MLflow experiment tracking directory
└── output/                        # Generated predictions and metrics
```

## Model Details

### Architecture

The model uses a scikit-learn pipeline with:

- **Preprocessing**: OneHotEncoding for nominal features, OrdinalEncoding for ordinal features, passthrough for numeric features
- **Model**: LightGBM classifier (500 estimators, learning rate 0.05)
- **Explainability**: SHAP TreeExplainer for feature importance

### Input Features

Required features for predictions:

- `Age`, `Income`, `LoanAmount`, `CreditScore`, `MonthsEmployed`, `NumCreditLines`, `LoanTerm` (numeric)
- `Education` (ordinal: High School, Bachelor's, Master's, PhD)
- `EmploymentType`, `MaritalStatus`, `LoanPurpose` (categorical)
- `HasMortgage`, `HasDependents`, `HasCoSigner` (binary: Yes/No)

### Performance Metrics

The model tracks:

- **ROC-AUC**: Area under the ROC curve
- **F1-Score**: Harmonic mean of precision and recall
- **Accuracy**: Proportion of correct predictions
- **Precision**: True positives / (true positives + false positives)
- **Recall**: True positives / (true positives + false negatives)
- **KS Statistic**: Maximum difference between cumulative distributions

## Dependencies

Key packages (see [requirements.txt](requirements.txt) for complete list):

- `streamlit>=1.28.0` - Web application framework
- `pandas>=2.0.0` - Data manipulation
- `numpy>=1.24.0` - Numerical computing
- `scikit-learn>=1.3.0` - ML preprocessing and metrics
- `lightgbm>=4.0.0` - Gradient boosting model
- `mlflow>=2.0.0` - Experiment tracking
- `shap>=0.50.0` - Feature importance explanation
- `matplotlib>=3.7.0`, `seaborn>=0.12.0` - Visualization

## Documentation

- [INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md) - Detailed architecture, integration flow, and component descriptions
- [Notebooks/](Notebooks/) - Jupyter notebooks with EDA, training, and analysis workflows

## Getting Help

### Common Issues

**Model not found error when running predictions:**

- Ensure a model has been trained first: `python src/train.py`
- Check that MLflow tracking URI is correctly set in `model_utils.py`

**SHAP explainer initialization warning:**

- The system automatically falls back to KernelExplainer if TreeExplainer fails
- This may occur with certain model pipeline configurations

**Missing input columns:**

- Verify your input data contains all required features (see Model Details section)
- Run `validate_input()` from `model_utils.py` to check your data

### Further Information

- **MLflow Tracking**: Model runs and artifacts are stored in `mlruns/` directory
- **Notebook Analysis**: See [Notebooks/](Notebooks/) for exploratory analysis and business cutoff analysis
- **Configuration**: Edit hyperparameters in `src/train.py` for model tuning

## Contributing

Contributions are welcome! Please:

1. Create a feature branch: `git checkout -b feature/your-feature`
2. Make your changes with clear commit messages
3. Ensure code follows the existing style
4. Test your changes thoroughly
5. Submit a pull request with a description of your changes

## License

This project is provided as-is for educational and research purposes.

## Project Status

This is an active project. See [todo.txt](todo.txt) for planned improvements including:

- Unit tests with pytest
- Type hints for all functions
- Proper logging infrastructure
- Enhanced configuration management
- API endpoints for production deployment

---

**Last Updated**: March 2026
