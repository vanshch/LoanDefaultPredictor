# import block
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline
import mlflow
import lightgbm as lgb
import numpy as np
import os

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)


# helper functions
def load_data(loc: str) -> pd.DataFrame:
    return pd.read_csv(loc)


def split(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X = df.drop("Default", axis=1)
    y = df["Default"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y, shuffle=True
    )

    return X_train, X_test, y_train, y_test


def transform(X_train, X_test, y_train, y_test):

    num_unscaled_port = ["Age", "MonthsEmployed", "NumCreditLines", "LoanTerm"]

    num_scaled_port = [
        "Income",
        "LoanAmount",
        "CreditScore",
        # "InterestRate",
        # "DTIRatio"
    ]

    ordinal_cols_port = ["Education"]
    nominal_cols_port = ["EmploymentType", "MaritalStatus", "LoanPurpose"]
    binary_cols_port = ["HasMortgage", "HasDependents", "HasCoSigner"]

    education_order = [["High School", "Bachelor's", "Master's", "PhD"]]

    ordinal_encoder = OrdinalEncoder(categories=education_order)
    binary_encoder = OrdinalEncoder(
        categories=[["No", "Yes"]] * 3  # No -> 0 and yes -> 1
    )

    preprocessor_tree_port = ColumnTransformer(
        transformers=[
            ("num_scaled", "passthrough", num_scaled_port),
            ("num_unscaled", "passthrough", num_unscaled_port),
            ("binary", binary_encoder, binary_cols_port),
            ("ordinal", ordinal_encoder, ordinal_cols_port),
            (
                "nominal",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                nominal_cols_port,
            ),
        ],
        remainder="drop",
        # verbose=True
    )

    return preprocessor_tree_port


def train(X_train, X_test, y_train, y_test, preprocessor_tree_port):
    with mlflow.start_run(run_name="Pipeline_LightGBM_port"):

        pipe_lgb = Pipeline(
            steps=[
                ("preprocess", preprocessor_tree_port),
                (
                    "model",
                    lgb.LGBMClassifier(
                        n_estimators=500,
                        learning_rate=0.05,
                        max_depth=-1,
                        num_leaves=31,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        objective="binary",
                        random_state=42,
                    ),
                ),
            ]
        )

        pipe_lgb.fit(X_train, y_train)

        y_train_proba = pipe_lgb.predict_proba(X_train)[:, 1]
        y_test_proba = pipe_lgb.predict_proba(X_test)[:, 1]

        train_auc = roc_auc_score(y_train, y_train_proba)
        test_auc = roc_auc_score(y_test, y_test_proba)

        # ---- MLflow logging ----
        mlflow.log_params(pipe_lgb.named_steps["model"].get_params())
        mlflow.log_metric("train_roc_auc", train_auc)
        mlflow.log_metric("test_roc_auc", test_auc)

        # Additional metrics
        y_test_pred = (y_test_proba >= 0.5).astype(int)
        mlflow.log_metric("test_precision", precision_score(y_test, y_test_pred))
        mlflow.log_metric("test_recall", recall_score(y_test, y_test_pred))
        mlflow.log_metric("test_f1", f1_score(y_test, y_test_pred))
        mlflow.log_metric("test_accuracy", accuracy_score(y_test, y_test_pred))

        # Confusion matrix
        cm = confusion_matrix(y_test, y_test_pred)
        cm_df = pd.DataFrame(
            cm, index=["Actual_0", "Actual_1"], columns=["Pred_0", "Pred_1"]
        )
        mlflow.log_dict(cm_df.to_dict(), "confusion_matrix_lgb_port.json")

        # KS statistic and ROC curve
        fpr, tpr, thresholds = roc_curve(y_test, y_test_proba)
        ks = np.max(tpr - fpr)
        mlflow.log_metric("test_ks", float(ks))
        roc_df = pd.DataFrame({"fpr": fpr, "tpr": tpr, "thresholds": thresholds})
        roc_df.to_csv("roc_curve_lgb_port.csv", index=False)
        mlflow.log_dict(roc_df.to_dict(), "roc_curve_lgb_port.json")

        mlflow.sklearn.log_model(pipe_lgb, artifact_path="model")

        # ---- Feature importance ----
        feature_names = pipe_lgb.named_steps["preprocess"].get_feature_names_out()

        fi = pd.DataFrame(
            {
                "feature": feature_names,
                "importance": pipe_lgb.named_steps["model"].feature_importances_,
            }
        ).sort_values("importance", ascending=False)

        fi.to_csv("lgb_feature_importance_port.csv", index=False)
        mlflow.log_dict(fi.to_dict(), "lgb_feature_importance_port.json")

        print("LightGBM Train ROC-AUC:", train_auc)
        print("LightGBM Test ROC-AUC:", test_auc)
        print("LightGBM Test KS:", ks)
        print(fi.head(10))


# main code block
def main():
    PATH_CSV = "Dataset/Loan_default.csv"
    df = load_data(PATH_CSV)

    proj_root_path = os.path.abspath(os.path.join(os.getcwd(), ".."))
    tracking_uri = os.path.join(proj_root_path, "mlruns")
    mlflow.set_tracking_uri(f"file:{tracking_uri}")
    mlflow.set_experiment("loan_default_experiment")

    X_train, X_test, y_train, y_test = split(df)

    preprocessor_tree_port = transform(X_train, X_test, y_train, y_test)

    train(X_train, X_test, y_train, y_test, preprocessor_tree_port)


# package
if __name__ == "__main__":
    main()
