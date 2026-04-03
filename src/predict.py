# import block
import pandas as pd
import numpy as np
import os
from pathlib import Path
from datetime import datetime
from model_utils import (
    load_test_data,
    validate_input,
    load_model,
    load_background_data,
    initialize_shap_explainer,
    compute_shap_values,
    predict,
    evaluate_predictions,
)

x_TEST_PATH = "Dataset/x_test.csv"
Y_TEST_PATH = "Dataset/y_test.csv"

def save_predictions(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    y_pred: np.ndarray,
    y_pred_proba: np.ndarray,
    metrics: dict,
    run_id: str,
    shap_values=None,
    feature_importance_df=None,
):
    """Save predictions, metrics, and SHAP explanations to output folder"""

    # Create output directory
    output_dir = Path(__file__).parent.parent / "output"
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create predictions dataframe
    predictions_df = X_test.copy()
    predictions_df["actual_default"] = y_test.values
    predictions_df["predicted_default"] = y_pred
    predictions_df["default_probability"] = y_pred_proba
    predictions_df["correct_prediction"] = (y_test.values == y_pred).astype(int)

    # Add top SHAP values for each prediction if available
    if shap_values is not None and feature_importance_df is not None:
        # Add the most important SHAP value for each sample
        top_features = feature_importance_df.head(5)["feature"].tolist()
        for i, feature in enumerate(top_features):
            if i < shap_values.shape[1]:
                predictions_df[f"shap_{feature}"] = shap_values[:, i]

    # Save predictions
    predictions_file = output_dir / f"predictions_{timestamp}.csv"
    predictions_df.to_csv(predictions_file, index=False)
    print(f"✓ Predictions saved to: {predictions_file}")

    # Save metrics
    metrics_df = pd.DataFrame([metrics])
    metrics_df["timestamp"] = timestamp
    metrics_df["model_run_id"] = run_id
    metrics_file = output_dir / f"metrics_{timestamp}.csv"
    metrics_df.to_csv(metrics_file, index=False)
    print(f"✓ Metrics saved to: {metrics_file}")

    # Save SHAP feature importance if available
    if feature_importance_df is not None:
        shap_file = output_dir / f"shap_feature_importance_{timestamp}.csv"
        feature_importance_df.to_csv(shap_file, index=False)
        print(f"✓ SHAP feature importance saved to: {shap_file}")

    # Save summary report
    summary_file = output_dir / f"prediction_summary_{timestamp}.txt"
    with open(summary_file, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("LOAN DEFAULT PREDICTION SUMMARY\n")
        f.write("=" * 60 + "\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Model Run ID: {run_id}\n")
        f.write(f"Total Predictions: {len(y_pred)}\n")
        f.write(
            f"Predicted Defaults: {y_pred.sum()} ({y_pred.sum()/len(y_pred)*100:.2f}%)\n"
        )
        f.write(
            f"Actual Defaults: {y_test.sum()} ({y_test.sum()/len(y_test)*100:.2f}%)\n"
        )
        f.write("\n" + "=" * 60 + "\n")
        f.write("PERFORMANCE METRICS\n")
        f.write("=" * 60 + "\n")
        for metric_name, value in metrics.items():
            f.write(f"{metric_name.upper():<20}: {value:.4f}\n")

        # Add SHAP feature importance to summary
        if feature_importance_df is not None:
            f.write("\n" + "=" * 60 + "\n")
            f.write("TOP 10 FEATURE IMPORTANCE (SHAP VALUES)\n")
            f.write("=" * 60 + "\n")
            for idx, row in feature_importance_df.head(10).iterrows():
                f.write(f"{row['feature']:<30}: {row['mean_abs_shap_value']:.6f}\n")

    print(f"✓ Summary report saved to: {summary_file}")

    return predictions_file, metrics_file, summary_file


# main code block
def main():
    """Main function to run loan default predictions with SHAP explainability"""

    print("\n" + "=" * 60)
    print("LOAN DEFAULT PREDICTION PIPELINE WITH SHAP")
    print("=" * 60 + "\n")

    # 1. Load test data
    print("Step 1: Loading test data...")
    X_test, y_test = load_test_data(x_TEST_PATH, Y_TEST_PATH)

    # 2. Validate input data
    print("\nStep 2: Validating input data...")
    validate_input(X_test)

    # 3. Load trained model
    print("\nStep 3: Loading trained model from MLflow...")
    model, run_id = load_model()

    # 4. Load background data for SHAP
    print("\nStep 4: Loading background data for SHAP explainer...")
    X_background = load_background_data(sample_size=100)

    # 5. Initialize SHAP explainer
    print("\nStep 5: Initializing SHAP explainer...")
    explainer, X_background_transformed = initialize_shap_explainer(model, X_background)

    # 6. Run predictions
    print("\nStep 6: Running predictions...")
    y_pred, y_pred_proba = predict(model, X_test)

    # 7. Compute SHAP values for explainability
    print("\nStep 7: Computing SHAP values for feature importance...")
    shap_values, feature_importance_df = compute_shap_values(
        explainer, model, X_test, X_background_transformed
    )

    # 8. Evaluate predictions
    print("\nStep 8: Evaluating predictions...")
    metrics, cm = evaluate_predictions(y_test, y_pred, y_pred_proba)

    # Print confusion matrix
    print("\n" + "=" * 50)
    print("MODEL PERFORMANCE METRICS")
    print("=" * 50)
    for metric_name, value in metrics.items():
        print(f"{metric_name.upper():<20}: {value:.4f}")

    print("\nCONFUSION MATRIX:")
    print(f"{'':>15} Predicted 0  Predicted 1")
    print(f"Actual 0        {cm[0][0]:>10}  {cm[0][1]:>10}")
    print(f"Actual 1        {cm[1][0]:>10}  {cm[1][1]:>10}")
    print("=" * 50 + "\n")

    # 9. Save outputs
    print("\nStep 9: Saving outputs...")
    predictions_file, metrics_file, summary_file = save_predictions(
        X_test,
        y_test,
        y_pred,
        y_pred_proba,
        metrics,
        run_id,
        shap_values,
        feature_importance_df,
    )

    print("\n" + "=" * 60)
    print("PREDICTION PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 60 + "\n")


# package
if __name__ == "__main__":
    main()
