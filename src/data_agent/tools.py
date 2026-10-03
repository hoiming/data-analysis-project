"""Tools the analyst agent can call. Arguments stay structured on purpose."""

from __future__ import annotations

import json
from typing import Any

from langchain.tools import ToolRuntime, tool

from data_agent.charts import plot_chart as render_chart
from data_agent.ml import train_model
from data_agent.datasets import (
    DatasetError,
    aggregate,
    correlate,
    describe_dataset as describe_table,
    list_dataset_files,
    query_rows,
    value_counts,
)

_MAX_CHARS = 12_000


def build_tools() -> list[Any]:
    return [
        list_datasets,
        describe_dataset,
        query_dataset,
        aggregate_dataset,
        count_values,
        correlate_numeric,
        plot_dataset,
        train_ml_model,
        save_finding,
        save_preference,
        search_memory,
    ]


@tool
def list_datasets() -> str:
    """List tabular files available in the data folder, with row and column counts."""
    return _ok(list_dataset_files())


@tool
def describe_dataset(name: str, sheet_name: str = "") -> str:
    """Describe one dataset: columns, types, missing values, a numeric summary, and a short preview.

    sheet_name is optional and only used for Excel files.
    """
    return _guard(lambda: describe_table(name, sheet_name=_sheet(sheet_name)))


@tool
def query_dataset(
    name: str,
    filters_json: str = "[]",
    columns_json: str = "[]",
    sort_by: str = "",
    ascending: bool = True,
    limit: int = 30,
    sheet_name: str = "",
) -> str:
    """Return rows from a dataset.

    filters_json is a JSON list of objects: {"column", "op", "value"}.
    op is eq, ne, gt, gte, lt, lte, contains, not_contains, or in.
    columns_json is a JSON list of column names to keep. Leave it empty for every column.
    """
    return _guard(
        lambda: query_rows(
            name,
            filters=_object_list(filters_json, "filters_json"),
            columns=_string_list(columns_json) or None,
            sort_by=sort_by.strip() or None,
            ascending=ascending,
            limit=limit,
            sheet_name=_sheet(sheet_name),
        )
    )


@tool
def aggregate_dataset(
    name: str,
    metrics_json: str,
    group_by_json: str = "[]",
    filters_json: str = "[]",
    sheet_name: str = "",
) -> str:
    """Aggregate a dataset.

    metrics_json is a JSON list. Each object has "agg" and usually "column".
    agg is count, sum, mean, min, max, nunique, median, or sum_product.
    sum_product takes "columns": ["left", "right"] and sums left * right, for example units * unit_price.
    group_by_json is a JSON list of column names. Omit it for a single overall result.
    """
    return _guard(
        lambda: aggregate(
            name,
            metrics=_object_list(metrics_json, "metrics_json"),
            group_by=_string_list(group_by_json) or None,
            filters=_object_list(filters_json, "filters_json"),
            sheet_name=_sheet(sheet_name),
        )
    )


@tool
def count_values(name: str, column: str, filters_json: str = "[]", limit: int = 20, sheet_name: str = "") -> str:
    """Count how often each value appears in one column."""
    return _guard(
        lambda: value_counts(
            name,
            column,
            filters=_object_list(filters_json, "filters_json"),
            limit=limit,
            sheet_name=_sheet(sheet_name),
        )
    )


@tool
def correlate_numeric(name: str, columns_json: str = "[]", sheet_name: str = "") -> str:
    """Pearson correlation for numeric columns. columns_json limits which columns are included."""
    return _guard(
        lambda: correlate(
            name,
            columns=_string_list(columns_json) or None,
            sheet_name=_sheet(sheet_name),
        )
    )


@tool
def plot_dataset(
    name: str,
    chart_type: str,
    x: str = "",
    y: str = "",
    title: str = "",
    filters_json: str = "[]",
    sheet_name: str = "",
) -> str:
    """Save a chart and return its file path. chart_type is bar, line, scatter, hist, or box.

    bar, line, and scatter need x and y. hist and box need one numeric column in x or y.
    The chat UI displays the image from the returned path.
    """
    return _guard(
        lambda: render_chart(
            name,
            chart_type,
            x=x.strip() or None,
            y=y.strip() or None,
            title=title.strip() or None,
            filters=_object_list(filters_json, "filters_json"),
            sheet_name=_sheet(sheet_name),
        )
    )


@tool
def train_ml_model(
    name: str,
    target: str,
    method: str,
    features_json: str = "[]",
    task: str = "auto",
    test_size: float = 0.25,
    n_neighbors: int = 5,
    sheet_name: str = "",
) -> str:
    """Fit a scikit-learn model and return held-out metrics.

    method is linear_regression, svm, or knn.
    linear_regression predicts a numeric target.
    svm and knn use task "regression" or "classification". task "auto" classifies when the target has few distinct labels, otherwise it regresses.
    features_json is a JSON list of feature columns. Leave it empty to use every column except the target.
    n_neighbors applies only to knn.
    """
    return _guard(
        lambda: train_model(
            name,
            target,
            method,
            features=_string_list(features_json) or None,
            task=task,
            test_size=test_size,
            n_neighbors=n_neighbors,
            sheet_name=_sheet(sheet_name),
        )
    )


@tool
def save_finding(title: str, finding: str, dataset: str = "", *, runtime: ToolRuntime) -> str:
    """Save a conclusion so later conversations can retrieve it by meaning."""
    key = store_save_finding(_store(runtime), title, finding, dataset)
    return f"Saved finding {key}."


@tool
def save_preference(preference: str, *, runtime: ToolRuntime) -> str:
    """Save a lasting user preference, such as a preferred metric, chart type, or rounding."""
    key = _put_preference(_store(runtime), preference)
    return f"Saved preference {key}."


@tool
def search_memory(query: str, *, runtime: ToolRuntime) -> str:
    """Search findings and preferences from earlier conversations by meaning."""
    store = _store(runtime)
    findings = store.search(("findings",), query=query, limit=5)
    preferences = store.search(("preferences",), query=query, limit=3)
    rows = []
    for item in list(findings) + list(preferences):
        value = item.value or {}
        rows.append(
            {
                "namespace": "/".join(item.namespace),
                "title": value.get("title", ""),
                "text": value.get("text", ""),
                "score": item.score,
            }
        )
    if not rows:
        return "No matching memories."
    return _ok(rows)


def store_save_finding(store: Any, title: str, finding: str, dataset: str = "") -> str:
    from data_agent.memory import put_finding

    return put_finding(store, title, finding, dataset)


def _put_preference(store: Any, preference: str) -> str:
    from data_agent.memory import put_preference

    return put_preference(store, preference)


def _store(runtime: ToolRuntime | None) -> Any:
    if runtime is None or runtime.store is None:
        raise DatasetError("Long-term memory is not available.")
    return runtime.store


def _guard(func: Any) -> str:
    try:
        return _ok(func())
    except (DatasetError, json.JSONDecodeError, ValueError, TypeError) as exc:
        return f"Error: {exc}"


def _ok(payload: Any) -> str:
    text = json.dumps(payload, ensure_ascii=False, default=str)
    if len(text) > _MAX_CHARS:
        return text[:_MAX_CHARS] + "...(truncated)"
    return text


def _sheet(value: str) -> str | int | None:
    text = (value or "").strip()
    if not text:
        return None
    if text.isdigit():
        return int(text)
    return text


def _loads(value: str, label: str) -> Any:
    text = (value or "").strip()
    if not text:
        return []
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise DatasetError(f"{label} must be JSON. {exc.msg}.") from exc


def _object_list(value: str, label: str) -> list[dict[str, Any]]:
    parsed = _loads(value, label)
    if parsed is None:
        return []
    if isinstance(parsed, dict):
        parsed = [parsed]
    if not isinstance(parsed, list) or any(not isinstance(item, dict) for item in parsed):
        raise DatasetError(f"{label} must be a JSON list of objects.")
    return parsed


def _string_list(value: str) -> list[str]:
    text = (value or "").strip()
    if not text:
        return []
    if text.startswith("["):
        parsed = _loads(text, "columns")
        if isinstance(parsed, str):
            return [parsed]
        if not isinstance(parsed, list):
            raise DatasetError("Expected a JSON list of column names.")
        return [str(item) for item in parsed]
    return [part.strip() for part in text.split(",") if part.strip()]
