#import block
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline
import lightgbm as lgb

from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    accuracy_score, confusion_matrix, roc_auc_score,
    roc_curve
)

# helper functions
def load_data(loc:str) -> pd.Dataframe:
    return pd.read_csv(loc)

def split(df:pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    X = df.drop('Default',axis = 1)
    y = df['Default']

    X_train,X_temp,y_train,y_temp= train_test_split(
        X,
        y,
        test_size = 0.2,
        random_state = 42,
        stratify=y,
        shuffle = True
    )

    X_val,X_test,y_val,y_test= train_test_split(
        X_temp,
        y_temp,
        test_size = 0.5, # equal split among the test
        random_state = 42,
        stratify=y,
        shuffle = True
    )

    return X_train,X_val,X_test,y_train,y_val,y_test

    

def transform(X_train,X_val,X_test,y_train,y_val,y_test):
    
    num_unscaled_port = [
        "Age",
        "MonthsEmployed",
        "NumCreditLines",
        "LoanTerm"
    ]

    num_scaled_port = [
        "Income",
        "LoanAmount",
        "CreditScore",
        # "InterestRate",
        # "DTIRatio"
    ]

    ordinal_cols_port = [
        "Education"
    ]
    nominal_cols_port = [
        "EmploymentType",
        "MaritalStatus",
        "LoanPurpose"
    ]
    binary_cols_port = [
        "HasMortgage",
        "HasDependents",
        "HasCoSigner"
    ]

    education_order = [
        ["High School", "Bachelor's", "Master's", "PhD"]
    ]

    ordinal_encoder = OrdinalEncoder(categories=education_order)
    binary_encoder = OrdinalEncoder(
        categories=[['No','Yes']] * 3 # No -> 0 and yes -> 1
    )

    preprocessor_tree_port = ColumnTransformer(
        transformers=[
            ("num_scaled", "passthrough", num_scaled_port),
            ("num_unscaled", "passthrough", num_unscaled_port),
            ("binary",binary_encoder, binary_cols_port),
            ("ordinal", ordinal_encoder, ordinal_cols_port),
            ("nominal", OneHotEncoder(handle_unknown="ignore", sparse_output=False), nominal_cols_port),
        ],
        remainder="drop",
        # verbose=True
    )

    return preprocessor_tree_port

def train(X_train,X_val,X_test,y_train,y_val,y_test,preprocessor_tree_port):
    pipe_lgb = Pipeline(
        steps=[
            ("preprocess",preprocessor_tree_port ),
            ("model", lgb.LGBMClassifier(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=-1,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="binary",
                random_state=42
            ))
        ]
    )

    pipe_lgb.fit(X_train, y_train)

    y_train_proba = pipe_lgb.predict_proba(X_train)[:, 1]
    y_test_proba = pipe_lgb.predict_proba(X_test)[:, 1]

    train_auc = roc_auc_score(y_train, y_train_proba)
    test_auc = roc_auc_score(y_test, y_test_proba)




# main code block 
def main():
    pass


# package 
if __name__ == "__main__":
    main()