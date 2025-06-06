import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    StratifiedKFold,
)
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

# Third-party boosting libraries
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Visualization
import seaborn as sns
import matplotlib.pyplot as plt

# Utility path for saving
SAVE_DIR = Path("results")
SAVE_DIR.mkdir(exist_ok=True)

df_ml = # Import dataset

# Feature/target split
X = df_ml.drop(columns=["target"])
y = df_ml["target"]

# Feature typing
categorical_feats = X.select_dtypes(include=["object", "category"]).columns.tolist()
numeric_feats = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

# Preprocessors
numeric_transformer = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)
categorical_transformer = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)
preprocessor = ColumnTransformer(
    [
        ("num", numeric_transformer, numeric_feats),
        ("cat", categorical_transformer, categorical_feats),
    ]
)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Define models
models = {
    "LogisticRegression": LogisticRegression(
        max_iter=2000, class_weight="balanced", random_state=42, verbose=1
    ),
    "RandomForest": RandomForestClassifier(
        n_estimators=200, class_weight="balanced", random_state=42, verbose=1
    ),
    "XGBoost": XGBClassifier(eval_metric="mlogloss", random_state=42, verbosity=1),
    "LightGBM": LGBMClassifier(random_state=42),
    "CatBoost": CatBoostClassifier(verbose=1, random_state=42),
}

# Add stacking
stacking = StackingClassifier(
    estimators=[
        (name, models[name]) for name in ["RandomForest", "LightGBM", "XGBoost"]
    ],
    final_estimator=LogisticRegression(),
    cv=5,
)
models["Stacking"] = stacking

# Cross-validation
cv_results = []
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    pipe = Pipeline([("preproc", preprocessor), ("clf", model)])
    scores = cross_val_score(
        pipe,
        X_train,
        y_train,
        cv=skf,
        scoring="accuracy",
        n_jobs=-1,
    )
    cv_results.append(
        {
            "model": name,
            "cv_mean_accuracy": np.mean(scores),
            "cv_std_accuracy": np.std(scores),
        }
    )

cv_df = pd.DataFrame(cv_results).sort_values("cv_mean_accuracy", ascending=False)
cv_df.to_csv(SAVE_DIR / "cv_results.csv", index=False)

# Evaluate all models on test set
full_results = []

for name, model in models.items():
    print(f"Training and evaluating: {name}")
    pipe = Pipeline([("preproc", preprocessor), ("clf", model)])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    try:
        y_proba = pipe.predict_proba(X_test)
        roc_auc = roc_auc_score(y_test, y_proba, multi_class="ovr")
    except:
        y_proba = None
        roc_auc = np.nan

    full_results.append(
        {
            "model": name,
            "accuracy": accuracy_score(y_test, y_pred),
            "f1_macro": f1_score(y_test, y_pred, average="macro"),
            "f1_weighted": f1_score(y_test, y_pred, average="weighted"),
            "precision_macro": precision_score(y_test, y_pred, average="macro"),
            "recall_macro": recall_score(y_test, y_pred, average="macro"),
            "roc_auc_ovr": roc_auc,
        }
    )

    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).T
    report_df.to_csv(SAVE_DIR / f"{name}_classification_report.csv")
    joblib.dump(pipe, SAVE_DIR / f"{name}_pipeline.joblib")

metrics_df = pd.DataFrame(full_results)
metrics_df.to_csv(SAVE_DIR / "test_set_metrics.csv", index=False)

