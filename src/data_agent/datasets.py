"""Safe loading and tabular operations for files in the data folder.

The model never receives a path it can use to leave `data_dir`, and it never
evaluates a Python expression against a frame. Filters and aggregations are
structured arguments applied here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from data_agent.config import settings

SUPPORTED_SUFFIXES = {".csv", ".xlsx", ".xls"}
FilterOp = Literal["eq", "ne", "gt", "gte", "lt", "lte", "contains", "not_contains", "in"]
AggName = Literal["count", "sum", "mean", "min", "max", "nunique", "median"]

_CACHE: dict[tuple[str, float, int], pd.DataFrame] = {}


class DatasetError(ValueError):
    """A user-facing problem with a dataset name or an operation."""


def list_dataset_files() -> list[dict[str, Any]]:
    files = []
    for path in sorted(settings.data_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        if path.name.startswith("."):
            continue
        frame = load_dataset(path.name)
        files.append(
            {
                "name": path.name,
                "rows": int(len(frame)),
                "columns": int(len(frame.columns)),
                "bytes": path.stat().st_size,
            }
        )
    return files


def resolve_dataset(name: str) -> Path:
    if not name or name.strip() != Path(name).name:
        raise DatasetError("Use a file name from the data folder, not a path.")
    path = (settings.data_dir / name).resolve()
    if path.parent != settings.data_dir:
        raise DatasetError("That file is outside the data folder.")
    if path.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise DatasetError(
            f"Unsupported file type '{path.suffix}'. Use a .csv, .xlsx, or .xls file."
        )
    if not path.is_file():
        known = ", ".join(item["name"] for item in _names_only()) or "(none)"
        raise DatasetError(f"No dataset named '{name}'. Available: {known}")
    return path


def load_dataset(name: str, sheet_name: str | int | None = None) -> pd.DataFrame:
    path = resolve_dataset(name)
    key = (str(path), path.stat().st_mtime, 0 if sheet_name is None else hash(str(sheet_name)))
    cached = _CACHE.get(key)
    if cached is not None:
        return cached.copy()

    suffix = path.suffix.lower()
    if suffix == ".csv":
        frame = pd.read_csv(path)
    elif suffix in {".xlsx", ".xls"}:
        frame = pd.read_excel(path, sheet_name=0 if sheet_name is None else sheet_name)
    else:
        raise DatasetError(f"Unsupported file type '{suffix}'. Use a .csv, .xlsx, or .xls file.")

    if not isinstance(frame, pd.DataFrame):
        raise DatasetError("That file did not load as a single table.")
    _CACHE[key] = frame
    return frame.copy()


def describe_dataset(name: str, sheet_name: str | int | None = None) -> dict[str, Any]:
    frame = load_dataset(name, sheet_name=sheet_name)
    columns = []
    for column in frame.columns:
        series = frame[column]
        non_null = series.dropna()
        sample = [str(value) for value in non_null.head(3).tolist()]
        columns.append(
            {
                "name": str(column),
                "dtype": str(series.dtype),
                "missing": int(series.isna().sum()),
                "nunique": int(series.nunique(dropna=True)),
                "sample": sample,
            }
        )
    numeric = frame.select_dtypes(include="number")
    summary = {}
    if not numeric.empty:
        summary = json.loads(numeric.describe().round(4).to_json())
    return {
        "name": name,
        "rows": int(len(frame)),
        "column_count": int(len(frame.columns)),
        "columns": columns,
        "numeric_summary": summary,
        "preview": _records(frame.head(5)),
    }


def apply_filters(frame: pd.DataFrame, filters: list[dict[str, Any]] | None) -> pd.DataFrame:
    result = frame
    for item in filters or []:
        column = _require_column(result, item.get("column"))
        op: str = str(item.get("op", "eq"))
        value = item.get("value")
        series = result[column]
        mask = _mask(series, op, value)
        result = result.loc[mask]
    return result


def query_rows(
    name: str,
    filters: list[dict[str, Any]] | None = None,
    columns: list[str] | None = None,
    sort_by: str | None = None,
    ascending: bool = True,
    limit: int = 30,
    sheet_name: str | int | None = None,
) -> dict[str, Any]:
    frame = apply_filters(load_dataset(name, sheet_name=sheet_name), filters)
    if columns:
        missing = [column for column in columns if column not in frame.columns]
        if missing:
            raise DatasetError(f"Unknown columns: {', '.join(missing)}")
        frame = frame.loc[:, columns]
    if sort_by:
        column = _require_column(frame, sort_by)
        frame = frame.sort_values(column, ascending=ascending, kind="mergesort")
    limit = _clamp_limit(limit)
    return {
        "name": name,
        "matched_rows": int(len(frame)),
        "returned_rows": int(min(len(frame), limit)),
        "rows": _records(frame.head(limit)),
    }


def aggregate(
    name: str,
    metrics: list[dict[str, Any]],
    group_by: list[str] | None = None,
    filters: list[dict[str, Any]] | None = None,
    limit: int = 30,
    sheet_name: str | int | None = None,
) -> dict[str, Any]:
    if not metrics:
        raise DatasetError("Provide at least one metric, for example sum of a numeric column.")
    frame = apply_filters(load_dataset(name, sheet_name=sheet_name), filters)
    group_by = group_by or []
    for column in group_by:
        _require_column(frame, column)

    limit = _clamp_limit(limit)
    if not group_by:
        values = {}
        for metric in metrics:
            label, value = _overall_metric(frame, metric)
            values[label] = value
        return {"name": name, "grouped": False, "result": values, "matched_rows": int(len(frame))}

    rows: list[dict[str, Any]] = []
    grouped = frame.groupby(group_by, dropna=False, sort=False)
    for keys, part in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {column: _jsonable(key) for column, key in zip(group_by, keys)}
        for metric in metrics:
            label, value = _overall_metric(part, metric)
            row[label] = value
        rows.append(row)
        if len(rows) >= limit:
            break
    return {
        "name": name,
        "grouped": True,
        "group_by": group_by,
        "matched_rows": int(grouped.ngroups),
        "rows": rows,
    }


def value_counts(
    name: str,
    column: str,
    filters: list[dict[str, Any]] | None = None,
    limit: int = 20,
    sheet_name: str | int | None = None,
) -> dict[str, Any]:
    frame = apply_filters(load_dataset(name, sheet_name=sheet_name), filters)
    column = _require_column(frame, column)
    limit = _clamp_limit(limit, upper=50)
    counts = frame[column].value_counts(dropna=False).head(limit)
    rows = [{"value": _jsonable(index), "count": int(count)} for index, count in counts.items()]
    return {"name": name, "column": column, "rows": rows}


def correlate(
    name: str,
    columns: list[str] | None = None,
    sheet_name: str | int | None = None,
) -> dict[str, Any]:
    frame = load_dataset(name, sheet_name=sheet_name)
    if columns:
        for column in columns:
            _require_column(frame, column)
        numeric = frame.loc[:, columns].select_dtypes(include="number")
    else:
        numeric = frame.select_dtypes(include="number")
    if numeric.shape[1] < 2:
        raise DatasetError("Correlation needs at least two numeric columns.")
    matrix = numeric.corr(numeric_only=True).round(4)
    return {"name": name, "columns": list(matrix.columns), "matrix": json.loads(matrix.to_json())}


def _names_only() -> list[dict[str, str]]:
    return [
        {"name": path.name}
        for path in sorted(settings.data_dir.iterdir())
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES and not path.name.startswith(".")
    ]


def _require_column(frame: pd.DataFrame, column: Any) -> str:
    if column not in frame.columns:
        known = ", ".join(str(name) for name in frame.columns)
        raise DatasetError(f"Unknown column '{column}'. Columns: {known}")
    return str(column)


def _mask(series: pd.Series, op: str, value: Any) -> pd.Series:
    if op == "eq":
        return series == value
    if op == "ne":
        return series != value
    if op == "in":
        if not isinstance(value, list):
            raise DatasetError("The 'in' filter expects a list of values.")
        return series.isin(value)
    if op in {"contains", "not_contains"}:
        text = series.astype("string")
        matched = text.str.contains(str(value), case=False, na=False, regex=False)
        return ~matched if op == "not_contains" else matched
    if op in {"gt", "gte", "lt", "lte"}:
        left = pd.to_numeric(series, errors="coerce")
        try:
            right = float(value)
        except (TypeError, ValueError) as exc:
            raise DatasetError(f"Comparison '{op}' needs a numeric value.") from exc
        comparators = {">": left.gt, ">=": left.ge, "<": left.lt, "<=": left.le}
        symbols = {"gt": ">", "gte": ">=", "lt": "<", "lte": "<="}
        return comparators[symbols[op]](right)
    raise DatasetError(
        "Unknown filter op. Use eq, ne, gt, gte, lt, lte, contains, not_contains, or in."
    )


def _overall_metric(frame: pd.DataFrame, metric: dict[str, Any]) -> tuple[str, Any]:
    agg = _agg_name(metric)
    if agg == "sum_product":
        columns = metric.get("columns") or []
        if len(columns) != 2:
            raise DatasetError("sum_product needs a 'columns' list of two numeric columns.")
        left_name = _require_column(frame, columns[0])
        right_name = _require_column(frame, columns[1])
        left = pd.to_numeric(frame[left_name], errors="coerce")
        right = pd.to_numeric(frame[right_name], errors="coerce")
        total = (left * right).sum(min_count=1)
        return f"sum_product_{left_name}_{right_name}", _jsonable(total)
    if agg == "count":
        column = metric.get("column")
        if column:
            _require_column(frame, column)
            value = int(frame[column].count())
            return f"count_{column}", value
        return "count_rows", int(len(frame))
    column = _require_column(frame, metric.get("column"))
    series = pd.to_numeric(frame[column], errors="coerce") if agg != "nunique" else frame[column]
    value = getattr(series, agg)()
    return f"{agg}_{column}", _jsonable(value)


def _agg_name(metric: dict[str, Any]) -> str:
    agg = str(metric.get("agg", "")).lower()
    allowed = {"count", "sum", "mean", "min", "max", "nunique", "median", "sum_product"}
    if agg not in allowed:
        raise DatasetError(f"Unknown aggregation '{agg}'. Use {', '.join(sorted(allowed))}.")
    return agg


def _clamp_limit(limit: int, upper: int | None = None) -> int:
    upper = upper or settings.max_preview_rows
    try:
        value = int(limit)
    except (TypeError, ValueError):
        value = upper
    return max(1, min(value, upper))


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(frame.to_json(orient="records", date_format="iso"))


def _jsonable(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value
