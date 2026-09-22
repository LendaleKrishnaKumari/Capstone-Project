from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree


ANALYTICS_DIR = Path(__file__).resolve().parent
DATASET_PATH = ANALYTICS_DIR / "titanic.csv"
PIPELINE_PATH = ANALYTICS_DIR / "best_titanic_pipeline.joblib"


def load_data() -> pd.DataFrame:
    """Load the one committed Titanic CSV used for all modeling work."""
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Titanic dataset not found: {DATASET_PATH}")
    return pd.read_csv(DATASET_PATH)


def build_modeling_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Create the stratified train/test split before any preprocessing."""
    feature_columns = [
        "pclass", "sex", "age", "sibsp", "parch", "fare", "embarked",
    ]
    x_dataframe = df[feature_columns]
    y_series = df["survived"]

    x_train, x_test, y_train, y_test = train_test_split(
        x_dataframe,
        y_series,
        test_size=0.2,
        random_state=42,
        stratify=y_series,
    )
    print("Class balance in target:")
    print(y_series.value_counts(normalize=True).round(4).to_string())
    print("\nTrain target distribution:")
    print(y_train.value_counts(normalize=True).round(4).to_string())
    return x_train, x_test, y_train, y_test


def make_preprocessor() -> ColumnTransformer:
    """Build a preprocessing transformer for the mixed-type Titanic features."""
    numeric_columns = ["age", "sibsp", "parch", "fare"]
    categorical_columns = ["sex", "embarked"]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
    )


def train_classifier_model(model_name: str, model: object, x_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    """Train a pipeline with preprocessing and a classifier."""
    pipeline = Pipeline(
        steps=[
            ("preprocessor", make_preprocessor()),
            ("model", model),
        ]
    )
    pipeline.fit(x_train, y_train)
    return pipeline


def evaluate_model(model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series, preprocessor: ColumnTransformer | None = None) -> dict:
    """Return the classification metrics for a fitted model."""
    if preprocessor is not None:
        x_test_prepared = preprocessor.transform(x_test)
        predictions = model.predict(x_test_prepared)
        probability_predictions = model.predict_proba(x_test_prepared)[:, 1]
    else:
        predictions = model.predict(x_test)
        probability_predictions = model.predict_proba(x_test)[:, 1]

    fpr, tpr, _ = roc_curve(y_test, probability_predictions)
    roc_auc = auc(fpr, tpr)

    metrics = {
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "precision": round(float(precision_score(y_test, predictions, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, predictions, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, predictions, zero_division=0)), 4),
        "auc": round(float(roc_auc), 4),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
    }

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.title("ROC curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(ANALYTICS_DIR / f"{model.__class__.__name__}_roc.png", dpi=150)
    plt.close()

    return metrics


def compare_imbalance_strategies(x_train: pd.DataFrame, y_train: pd.Series, x_test: pd.DataFrame, y_test: pd.Series) -> None:
    """Compare baseline, class_weight-balanced, and SMOTE for one classifier."""
    preprocessor = make_preprocessor()
    x_train_prepared = preprocessor.fit_transform(x_train)
    x_test_prepared = preprocessor.transform(x_test)

    baseline_model = LogisticRegression(max_iter=1000, random_state=42)
    baseline_model.fit(x_train_prepared, y_train)

    balanced_model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    balanced_model.fit(x_train_prepared, y_train)

    smote = SMOTE(random_state=42)
    x_train_smote, y_train_smote = smote.fit_resample(x_train_prepared, y_train)
    smote_model = LogisticRegression(max_iter=1000, random_state=42)
    smote_model.fit(x_train_smote, y_train_smote)

    for name, model in [("baseline", baseline_model), ("class_weight_balanced", balanced_model), ("smote", smote_model)]:
        metrics = evaluate_model(model, x_test, y_test, preprocessor=preprocessor)
        print(f"\n{name}: accuracy={metrics['accuracy']}, precision={metrics['precision']}, recall={metrics['recall']}, f1={metrics['f1']}")


def hyperparameter_tuning(x_train: pd.DataFrame, y_train: pd.Series) -> None:
    """Run GridSearchCV for the Random Forest model and report the best OOB score."""
    estimator = RandomForestClassifier(oob_score=True, random_state=42)
    param_grid = {
        "model__n_estimators": [50, 100, 200],
        "model__max_depth": [3, 5, None],
        "model__max_features": ["sqrt", "log2", None],
    }

    pipeline = Pipeline(
        steps=[
            ("preprocessor", make_preprocessor()),
            ("model", estimator),
        ]
    )
    grid = GridSearchCV(pipeline, param_grid=param_grid, cv=3, n_jobs=-1)
    grid.fit(x_train, y_train)

    print("\nBest RandomForest params:", grid.best_params_)
    print("Best OOB score:", grid.best_estimator_.named_steps["model"].oob_score_)


def regression_side_task(df: pd.DataFrame) -> dict:
    """Train a linear regression model to predict fare and return regression metrics."""
    feature_columns = ["pclass", "sex", "age", "sibsp", "parch", "embarked"]
    x_df = df[feature_columns].copy()
    y_series = df["fare"]

    x_train, x_test, y_train, y_test = train_test_split(
        x_df,
        y_series,
        test_size=0.2,
        random_state=42,
    )

    numeric_columns = ["age", "sibsp", "parch"]
    categorical_columns = ["sex", "embarked"]
    numeric_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    categorical_pipeline = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OneHotEncoder(handle_unknown="ignore"))])

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
    )

    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("model", LinearRegression()),
    ])
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    residuals = y_test - predictions

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2_value = r2_score(y_test, predictions)
    n = len(y_test)
    p = x_train.shape[1]
    adj_r2 = 1 - (1 - r2_value) * (n - 1) / (n - p - 1)

    plt.figure(figsize=(6, 5))
    plt.scatter(predictions, residuals, alpha=0.6)
    plt.axhline(0, color="red", linestyle="--")
    plt.xlabel("Predicted fare")
    plt.ylabel("Residuals")
    plt.title("Residual plot")
    plt.tight_layout()
    plt.savefig(ANALYTICS_DIR / "regression_residuals.png", dpi=150)
    plt.close()

    print("\nRegression metrics:")
    print({
        "mae": round(float(mae), 4),
        "rmse": round(float(rmse), 4),
        "r2": round(float(r2_value), 4),
        "adjusted_r2": round(float(adj_r2), 4),
    })
    return {
        "mae": round(float(mae), 4),
        "rmse": round(float(rmse), 4),
        "r2": round(float(r2_value), 4),
        "adjusted_r2": round(float(adj_r2), 4),
    }


def save_full_pipeline(x_train: pd.DataFrame, y_train: pd.Series) -> None:
    """Save the complete preprocessing pipeline and estimator as a single object."""
    pipeline = Pipeline(
        steps=[
            ("preprocessor", make_preprocessor()),
            ("model", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )
    pipeline.fit(x_train, y_train)
    joblib.dump(pipeline, PIPELINE_PATH)
    print(f"\nSaved pipeline to: {PIPELINE_PATH}")

    reloaded = joblib.load(PIPELINE_PATH)
    sample_row = pd.DataFrame([
        {
            "pclass": 1,
            "sex": "female",
            "age": 28,
            "sibsp": 0,
            "parch": 0,
            "fare": 80.0,
            "embarked": "S",
        }
    ])
    print("Reloaded pipeline prediction:", reloaded.predict(sample_row)[0])


def main() -> None:
    """Run the full Titanic analytical workflow and save the final pipeline."""
    df = load_data()
    x_train, x_test, y_train, y_test = build_modeling_split(df)

    logistic_model = train_classifier_model("logistic", LogisticRegression(max_iter=1000, random_state=42), x_train, y_train)
    decision_tree_model = train_classifier_model("decision_tree", DecisionTreeClassifier(random_state=42), x_train, y_train)
    random_forest_model = train_classifier_model("random_forest", RandomForestClassifier(random_state=42, n_estimators=200), x_train, y_train)

    print("\nLogistic metrics:", evaluate_model(logistic_model, x_test, y_test))
    print("\nDecision tree metrics:", evaluate_model(decision_tree_model, x_test, y_test))
    print("\nRandom forest metrics:", evaluate_model(random_forest_model, x_test, y_test))

    compare_imbalance_strategies(x_train, y_train, x_test, y_test)
    hyperparameter_tuning(x_train, y_train)
    regression_side_task(df)
    save_full_pipeline(x_train, y_train)


if __name__ == "__main__":
    main()
