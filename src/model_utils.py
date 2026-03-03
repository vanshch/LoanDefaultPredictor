# Utility functions for model loading, prediction and SHAP explainability
import pandas as pd
import mlflow
import numpy as np
import os
import shap
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
)


def load_test_data(
    x_test_path: str, y_test_path: str
) -> tuple[pd.DataFrame, pd.Series]:
    """Load test dataset from CSV files"""
    try:
        X_test = pd.read_csv(x_test_path)
        y_test = pd.read_csv(y_test_path).squeeze()  # Convert to Series
        print(
            f"✓ Loaded test data: X_test shape {X_test.shape}, y_test shape {y_test.shape}"
        )
        return X_test, y_test
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Test data files not found: {e}")
    except Exception as e:
        raise Exception(f"Error loading test data: {e}")


def validate_input(X_test: pd.DataFrame) -> bool:
    """Validate input data has required columns and correct data types"""

    # Expected columns based on train.py
    expected_columns = [
        "Age",
        "Income",
        "LoanAmount",
        "CreditScore",
        "MonthsEmployed",
        "NumCreditLines",
        "LoanTerm",
        "Education",
        "EmploymentType",
        "MaritalStatus",
        "HasMortgage",
        "HasDependents",
        "LoanPurpose",
        "HasCoSigner",
    ]

    # Check for missing columns
    missing_cols = set(expected_columns) - set(X_test.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Check for null values
    null_counts = X_test.isnull().sum()
    if null_counts.sum() > 0:
        print(
            f"⚠ Warning: Found null values in columns:\n{null_counts[null_counts > 0]}"
        )

    # Check data types (basic validation)
    numeric_cols = [
        "Age",
        "Income",
        "LoanAmount",
        "CreditScore",
        "MonthsEmployed",
        "NumCreditLines",
        "LoanTerm",
    ]
    for col in numeric_cols:
        if col in X_test.columns:
            if not pd.api.types.is_numeric_dtype(X_test[col]):
                raise TypeError(
                    f"Column '{col}' should be numeric but is {X_test[col].dtype}"
                )

    print("Input validation passed")
    return True


def load_model(
    experiment_name: str = "loan_default_experiment",
    run_name: str = "Pipeline_LightGBM_port",
):
    """Load the trained model from MLflow"""
    try:
        # Set up MLflow tracking
        proj_root_path = os.path.abspath(os.path.join(os.getcwd(), ".."))
        tracking_uri = os.path.join(proj_root_path, "mlruns")
        mlflow.set_tracking_uri(f"file:{tracking_uri}")

        # Get the experiment
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            raise ValueError(f"Experiment '{experiment_name}' not found")

        # Search for runs with the specified name
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=f"attribute.run_name = '{run_name}'",
            order_by=["start_time DESC"],
            max_results=1,
        )

        if runs.empty:
            raise ValueError(
                f"No runs found with name '{run_name}' in experiment '{experiment_name}'"
            )

        run_id = runs.iloc[0].run_id
        model_uri = f"runs:/{run_id}/model"

        print(f"✓ Loading model from run: {run_id}")
        model = mlflow.sklearn.load_model(model_uri)
        print("✓ Model loaded successfully")

        return model, run_id

    except Exception as e:
        raise Exception(f"Error loading model: {e}")


def load_background_data(sample_size: int = 100) -> pd.DataFrame:
    """Load background dataset for SHAP explainer from the training dataset.
    Uses sampling for efficiency in live/online scenarios.

    Args:
        sample_size: Number of samples to use as background (default 100 for efficiency)

    Returns:
        Sampled background dataset
    """
    try:
        # Load the full dataset - handle relative paths from different locations
        dataset_path = "Dataset/Loan_default.csv"
        if not os.path.exists(dataset_path):
            # Try alternate path if running from different directory
            dataset_path = "../Dataset/Loan_default.csv"

        df = pd.read_csv(dataset_path)

        # Remove target column and ID column
        feature_cols = [col for col in df.columns if col not in ["Default", "LoanID"]]
        X_background = df[feature_cols]

        # Sample for efficiency (important for live usage)
        if len(X_background) > sample_size:
            X_background = X_background.sample(n=sample_size, random_state=42)

        print(f"✓ Loaded background data: {X_background.shape[0]} samples for SHAP")
        return X_background

    except Exception as e:
        raise Exception(f"Error loading background data: {e}")


def initialize_shap_explainer(model, X_background: pd.DataFrame):
    """Initialize SHAP TreeExplainer with background data.
    TreeExplainer is optimized for tree-based models like LightGBM.

    Args:
        model: Trained model (LightGBM pipeline)
        X_background: Background dataset for SHAP

    Returns:
        Initialized SHAP explainer and transformed background data
    """
    try:
        # For pipeline models, extract the final estimator
        if hasattr(model, "named_steps"):
            # It's a pipeline, get the final estimator
            final_estimator = model.steps[-1][1]
        else:
            final_estimator = model

        # Transform background data through pipeline if needed
        if hasattr(model, "named_steps"):
            X_background_transformed = model[:-1].transform(X_background)
        else:
            X_background_transformed = X_background

        # Initialize TreeExplainer (efficient for tree models)
        explainer = shap.TreeExplainer(final_estimator, X_background_transformed)
        print("✓ SHAP explainer initialized successfully")
        return explainer, X_background_transformed

    except Exception as e:
        print(
            f"⚠ Warning: Could not initialize TreeExplainer, falling back to KernelExplainer: {e}"
        )
        # Fallback to KernelExplainer if TreeExplainer fails
        try:
            explainer = shap.KernelExplainer(model.predict_proba, X_background)
            return explainer, X_background
        except Exception as e2:
            raise Exception(f"Error initializing SHAP explainer: {e2}")


def compute_shap_values(
    explainer, model, X_test: pd.DataFrame, X_background_transformed
):
    """Compute SHAP values for test data.

    Args:
        explainer: Initialized SHAP explainer
        model: Trained model
        X_test: Test dataset
        X_background_transformed: Transformed background data

    Returns:
        SHAP values array and feature importance dataframe
    """
    try:
        # Transform test data through pipeline if needed
        if hasattr(model, "named_steps"):
            X_test_transformed = model[:-1].transform(X_test)
        else:
            X_test_transformed = X_test

        # Compute SHAP values
        shap_values = explainer.shap_values(X_test_transformed)

        # For binary classification, get values for positive class
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Positive class (Default=1)

        # Compute mean absolute SHAP values for feature importance
        mean_abs_shap = np.abs(shap_values).mean(axis=0)

        # Get feature names from the transformed data
        feature_names = []
        if hasattr(X_test_transformed, "columns"):
            feature_names = X_test_transformed.columns.tolist()
        elif hasattr(model, "named_steps"):
            # Extract feature names from the pipeline
            try:
                # Try to get feature names from the preprocessing step
                preprocessor = model[:-1]
                if hasattr(preprocessor, "get_feature_names_out"):
                    feature_names = preprocessor.get_feature_names_out().tolist()
                else:
                    # Fallback: generate generic feature names based on shape
                    n_features = shap_values.shape[1]
                    feature_names = [f"feature_{i}" for i in range(n_features)]
            except:
                # Last resort: generic names
                n_features = shap_values.shape[1]
                feature_names = [f"feature_{i}" for i in range(n_features)]
        else:
            # Use original feature names if no transformation
            feature_names = X_test.columns.tolist()

        # Ensure feature_names matches the number of SHAP values
        if len(feature_names) != shap_values.shape[1]:
            n_features = shap_values.shape[1]
            feature_names = [f"feature_{i}" for i in range(n_features)]

        # Create feature importance dataframe
        feature_importance_df = pd.DataFrame(
            {"feature": feature_names, "mean_abs_shap_value": mean_abs_shap}
        ).sort_values("mean_abs_shap_value", ascending=False)

        print(f"✓ SHAP values computed for {len(X_test)} samples")

        return shap_values, feature_importance_df

    except Exception as e:
        raise Exception(f"Error computing SHAP values: {e}")


def predict(model, X_test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Run predictions on test data"""
    try:
        # Get probability predictions
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        # Get binary predictions (using 0.5 threshold)
        y_pred = (y_pred_proba >= 0.5).astype(int)

        print(f"✓ Predictions completed: {len(y_pred)} samples")
        return y_pred, y_pred_proba
    except Exception as e:
        raise Exception(f"Error during prediction: {e}")


def evaluate_predictions(
    y_test: pd.Series, y_pred: np.ndarray, y_pred_proba: np.ndarray
) -> dict:
    """Evaluate model performance on test data"""

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_pred_proba),
    }

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    return metrics, cm
