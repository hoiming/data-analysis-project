"""Save matplotlib charts under the artifacts folder and return their paths."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import uuid4

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from data_agent.config import settings
from data_agent.datasets import DatasetError, apply_filters, load_dataset

ChartType = {"bar", "line", "scatter", "hist", "box"}


def plot_chart(
    name: str,
    chart_type: str,
    x: str | None = None,
    y: str | None = None,
    title: str | None = None,
    filters: list[dict[str, Any]] | None = None,
    sheet_name: str | int | None = None,
) -> dict[str, str]:
    chart_type = chart_type.lower().strip()
    if chart_type not in ChartType:
        raise DatasetError(f"Unknown chart type '{chart_type}'. Use {', '.join(sorted(ChartType))}.")

    frame = apply_filters(load_dataset(name, sheet_name=sheet_name), filters)
    if frame.empty:
        raise DatasetError("Nothing to plot: the filters matched zero rows.")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    try:
        _draw(ax, frame, chart_type, x, y)
        ax.set_title(title or f"{chart_type} chart of {name}")
        ax.figure.tight_layout()
        path = settings.artifacts_dir / f"{uuid4().hex}.png"
        fig.savefig(path, dpi=120)
    finally:
        plt.close(fig)

    return {
        "path": str(path),
        "chart_type": chart_type,
        "dataset": name,
        "summary": f"Saved a {chart_type} chart to {path.name}.",
    }


def _draw(ax: plt.Axes, frame: pd.DataFrame, chart_type: str, x: str | None, y: str | None) -> None:
    if chart_type == "hist":
        column = _column(frame, x or y, "histogram column")
        series = pd.to_numeric(frame[column], errors="coerce").dropna()
        if series.empty:
            raise DatasetError(f"Column '{column}' has no numeric values to histogram.")
        ax.hist(series, bins=min(20, max(5, int(series.nunique()))))
        ax.set_xlabel(column)
        ax.set_ylabel("count")
        return

    if chart_type == "box":
        column = _column(frame, y or x, "box-plot column")
        series = pd.to_numeric(frame[column], errors="coerce").dropna()
        if series.empty:
            raise DatasetError(f"Column '{column}' has no numeric values to plot.")
        ax.boxplot(series, tick_labels=[column])
        ax.set_ylabel(column)
        return

    x_col = _column(frame, x, "x column")
    y_col = _column(frame, y, "y column")
    plot_frame = frame[[x_col, y_col]].dropna()
    if plot_frame.empty:
        raise DatasetError("The selected columns have no complete rows to plot.")
    if chart_type in {"line", "scatter"}:
        plot_frame[y_col] = pd.to_numeric(plot_frame[y_col], errors="coerce")
        plot_frame = plot_frame.dropna()
    if chart_type == "bar":
        ax.bar(plot_frame[x_col].astype(str), pd.to_numeric(plot_frame[y_col], errors="coerce"))
    elif chart_type == "line":
        ax.plot(plot_frame[x_col].astype(str), plot_frame[y_col])
    else:
        ax.scatter(pd.to_numeric(plot_frame[x_col], errors="coerce"), plot_frame[y_col])
        ax.set_xlabel(x_col)
    if chart_type != "scatter":
        ax.set_xlabel(x_col)
        ax.tick_params(axis="x", labelrotation=30)
    ax.set_ylabel(y_col)


def _column(frame: pd.DataFrame, name: str | None, role: str) -> str:
    if not name or name not in frame.columns:
        known = ", ".join(str(column) for column in frame.columns)
        raise DatasetError(f"A {role} is required. Columns: {known}")
    return name


def is_artifact(path: str) -> bool:
    candidate = Path(path)
    try:
        candidate.resolve().relative_to(settings.artifacts_dir)
    except ValueError:
        return False
    return candidate.is_file() and candidate.suffix.lower() == ".png"
