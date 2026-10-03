"""Held-out scikit-learn fits. The model chooses a method; it does not run arbitrary code."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC, SVR

from data_agent.datasets import DatasetError, load_dataset

METHODS = {"linear_regression", "svm", "knn"}
_MAX_ROWS = {"linear_regression": 8000, "knn": 8000, "svm": 3000}


def train_model(
    name: str,
    target: str,
    method: str,
    features: list[str] | None = None,
    task: str = "auto",
    test_size: float = 0.25,
    n_neighbors: int = 5,
    sheet_name: str | int | None = None,
) -> dict[str, Any]:
    method = method.strip().lower().replace(" ", "_").replace("-", "_")
    aliases = {
        "linear": "linear_regression",
        "linreg": "linear_regression",
        "regression": "linear_regression",
        "svr": "svm",
        "svc": "svm",
        "support_vector_machine": "svm",
        "k_nearest_neighbors": "knn",
        "k_neighbors": "knn",
    }
    method = aliases.get(method, method)
    if method not in METHODS:
        raise DatasetError("method must be linear_regression, svm, or knn.")

    frame = load_dataset(name, sheet_name=sheet_name)
    if target not in frame.columns:
        known = ", ".join(str(column) for column in frame.columns)
        raise DatasetError(f"Unknown target '{target}'. Columns: {known}")

    feature_names = _feature_names(frame, target, features)
    work = frame.loc[:, feature_names + [target]].copy()
    warnings: list[str] = []
    task_name, y = _prepare_target(work[target], method, task)
    work = work.loc[y.index]
    limit = _MAX_ROWS[method]
    if len(work) > limit:
        work = work.sample(limit, random_state=0)
        y = y.loc[work.index]
        warnings.append(f"Used a random sample of {limit} rows so {method} stays responsive.")

    if len(work) < 8:
        raise DatasetError("Need at least 8 rows with a target value to train a model.")

    x = work.loc[:, feature_names]
    for column in list(x.columns):
        if pd.api.types.is_datetime64_any_dtype(x[column]):
            x[column] = x[column].astype("int64")

    test_size = _test_size(test_size)
    stratify = y if task_name == "classification" and y.value_counts().min() >= 2 else None
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=0,
        stratify=stratify,
    )
    neighbors = _neighbors(n_neighbors, len(x_train))
    if method == "knn" and neighbors != int(n_neighbors):
        warnings.append(f"n_neighbors was reduced to {neighbors} to fit the training split.")

    pipeline = Pipeline(
        [
            ("preprocess", _preprocessor(x_train)),
            ("model", _estimator(method, task_name, neighbors)),
        ]
    )
    pipeline.fit(x_train, y_train)
    predicted = pipeline.predict(x_test)
    metrics = _metrics(task_name, y_test, predicted)
    preview = _preview(y_test, predicted, task_name)
    result: dict[str, Any] = {
        "dataset": name,
        "method": method,
        "task": task_name,
        "target": target,
        "features": feature_names,
        "rows_used": int(len(work)),
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "metrics": metrics,
        "preview": preview,
    }
    if method == "linear_regression":
        result["coefficients"] = _coefficients(pipeline)
    if method == "knn":
        result["n_neighbors"] = neighbors
    if warnings:
        result["warnings"] = warnings
    return result


def _feature_names(frame: pd.DataFrame, target: str, features: list[str] | None) -> list[str]:
    if not features:
        names = [str(column) for column in frame.columns if column != target]
    else:
        missing = [column for column in features if column not in frame.columns]
        if missing:
            raise DatasetError(f"Unknown feature columns: {', '.join(missing)}")
        names = [column for column in features if column != target]
    if not names:
        raise DatasetError("Choose at least one feature column other than the target.")
    return names


def _prepare_target(series: pd.Series, method: str, task: str) -> tuple[str, pd.Series]:
    requested = task.strip().lower()
    if requested not in {"auto", "regression", "classification"}:
        raise DatasetError("task must be auto, regression, or classification.")
    if method == "linear_regression" and requested == "classification":
        raise DatasetError("Linear regression predicts a numeric target. Use svm or knn to classify.")

    numeric = pd.to_numeric(series, errors="coerce")
    mostly_numeric = int(numeric.notna().sum()) >= max(8, int(len(series) * 0.8))
    if method == "linear_regression" or requested == "regression" or (
        requested == "auto" and mostly_numeric and int(numeric.nunique(dropna=True)) > 10
    ):
        if not mostly_numeric:
            raise DatasetError(f"Column '{series.name}' is not numeric enough for regression.")
        cleaned = numeric.dropna()
        return "regression", cleaned

    labels = series.dropna().astype(str)
    if labels.nunique() < 2:
        raise DatasetError("Classification needs at least two target classes.")
    if labels.nunique() > 30:
        raise DatasetError("That target has too many classes for svm or knn. Use regression, or pick a column with fewer labels.")
    return "classification", labels


def _preprocessor(frame: pd.DataFrame) -> ColumnTransformer:
    numeric = [column for column in frame.columns if pd.api.types.is_numeric_dtype(frame[column])]
    categorical = [column for column in frame.columns if column not in numeric]
    transformers = []
    if numeric:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric,
            )
        )
    if categorical:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical,
            )
        )
    if not transformers:
        raise DatasetError("The selected features have no usable values.")
    return ColumnTransformer(transformers)


def _estimator(method: str, task: str, n_neighbors: int):
    if method == "linear_regression":
        return LinearRegression()
    if method == "svm":
        return SVR(kernel="rbf", C=1.0) if task == "regression" else SVC(kernel="rbf", C=1.0)
    if task == "regression":
        return KNeighborsRegressor(n_neighbors=n_neighbors)
    return KNeighborsClassifier(n_neighbors=n_neighbors)


def _metrics(task: str, actual: pd.Series, predicted: np.ndarray) -> dict[str, float]:
    if task == "regression":
        return {
            "r2": _round(r2_score(actual, predicted)),
            "mae": _round(mean_absolute_error(actual, predicted)),
            "rmse": _round(float(np.sqrt(mean_squared_error(actual, predicted)))),
        }
    return {
        "accuracy": _round(accuracy_score(actual, predicted)),
        "f1_macro": _round(f1_score(actual, predicted, average="macro", zero_division=0)),
    }


def _coefficients(pipeline: Pipeline) -> list[dict[str, float | str]]:
    preprocess = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["model"]
    names = [str(name) for name in preprocess.get_feature_names_out()]
    values = np.ravel(model.coef_)
    ranked = sorted(zip(names, values), key=lambda item: abs(float(item[1])), reverse=True)
    return [{"feature": name, "coefficient": _round(value)} for name, value in ranked[:12]]


def _preview(actual: pd.Series, predicted: np.ndarray, task: str) -> list[dict[str, Any]]:
    rows = []
    for index, (truth, guess) in enumerate(zip(actual.tolist(), predicted.tolist())):
        if index >= 8:
            break
        if task == "regression":
            rows.append({"actual": _round(truth), "predicted": _round(guess)})
        else:
            rows.append({"actual": str(truth), "predicted": str(guess)})
    return rows


def _test_size(value: float) -> float:
    try:
        size = float(value)
    except (TypeError, ValueError) as exc:
        raise DatasetError("test_size must be a fraction between 0.1 and 0.4.") from exc
    if not 0.1 <= size <= 0.4:
        raise DatasetError("test_size must be between 0.1 and 0.4.")
    return size


def _neighbors(value: int, train_rows: int) -> int:
    try:
        neighbors = int(value)
    except (TypeError, ValueError) as exc:
        raise DatasetError("n_neighbors must be a positive integer.") from exc
    if neighbors < 1:
        raise DatasetError("n_neighbors must be at least 1.")
    return min(neighbors, max(1, train_rows - 1))


def _round(value: Any) -> float:
    return round(float(value), 4)
