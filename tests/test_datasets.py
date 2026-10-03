from pathlib import Path

import pandas as pd
import pytest

import os

from data_agent import config
from data_agent.charts import plot_chart
from data_agent.datasets import (
    DatasetError,
    aggregate,
    correlate,
    describe_dataset,
    load_dataset,
    query_rows,
    resolve_dataset,
    value_counts,
)


@pytest.fixture()
def data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    data = tmp_path / "data"
    artifacts = tmp_path / "artifacts"
    data.mkdir()
    artifacts.mkdir()
    frame = pd.DataFrame(
        {
            "region": ["North", "South", "North", "East"],
            "product": ["Widget", "Gadget", "Widget", "Gadget"],
            "units": [10, 4, 6, 8],
            "price": [25.0, 80.0, 25.0, 80.0],
        }
    )
    frame.to_csv(data / "sales.csv", index=False)
    monkeypatch.setattr(config.settings, "data_dir", data.resolve())
    monkeypatch.setattr(config.settings, "artifacts_dir", artifacts.resolve())
    return data


def test_socks_proxy_scheme_is_rewritten(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALL_PROXY", "socks://127.0.0.1:10808/")
    monkeypatch.setenv("https_proxy", "socks://127.0.0.1:10808")
    config.normalize_proxy_env()
    assert os.environ["ALL_PROXY"] == "socks5://127.0.0.1:10808/"
    assert os.environ["https_proxy"] == "socks5://127.0.0.1:10808"


def test_rejects_path_traversal(data_dir: Path) -> None:
    with pytest.raises(DatasetError):
        resolve_dataset("../secrets.csv")
    with pytest.raises(DatasetError):
        resolve_dataset("missing.csv")


def test_describe_and_filter(data_dir: Path) -> None:
    summary = describe_dataset("sales.csv")
    assert summary["rows"] == 4
    assert {column["name"] for column in summary["columns"]} == {"region", "product", "units", "price"}

    filtered = query_rows(
        "sales.csv",
        filters=[{"column": "region", "op": "eq", "value": "North"}],
        sort_by="units",
        ascending=False,
    )
    assert filtered["matched_rows"] == 2
    assert filtered["rows"][0]["units"] == 10


def test_aggregate_and_counts(data_dir: Path) -> None:
    grouped = aggregate(
        "sales.csv",
        group_by=["region"],
        metrics=[{"column": "units", "agg": "sum"}, {"column": "price", "agg": "mean"}],
    )
    north = next(row for row in grouped["rows"] if row["region"] == "North")
    assert north["sum_units"] == 16
    assert north["mean_price"] == 25

    counts = value_counts("sales.csv", "product")
    assert {row["value"]: row["count"] for row in counts["rows"]} == {"Widget": 2, "Gadget": 2}


def test_correlate_and_chart(data_dir: Path) -> None:
    matrix = correlate("sales.csv", columns=["units", "price"])
    assert set(matrix["columns"]) == {"units", "price"}

    chart = plot_chart("sales.csv", "bar", x="region", y="units", title="Units")
    assert Path(chart["path"]).is_file()
    assert chart["chart_type"] == "bar"


def test_sum_product_metric(data_dir: Path) -> None:
    result = aggregate(
        "sales.csv",
        metrics=[{"agg": "sum_product", "columns": ["units", "price"]}],
    )
    # 10*25 + 4*80 + 6*25 + 8*80 = 250 + 320 + 150 + 640
    assert result["result"]["sum_product_units_price"] == 1360


def test_query_tool_rejects_bad_json(data_dir: Path) -> None:
    from data_agent.tools import query_dataset

    result = query_dataset.invoke({"name": "sales.csv", "filters_json": "not-json"})
    assert result.startswith("Error:")


def test_contains_filter_is_literal(data_dir: Path) -> None:
    frame = load_dataset("sales.csv")
    assert len(frame) == 4
    matched = query_rows(
        "sales.csv",
        filters=[{"column": "product", "op": "contains", "value": "Wid"}],
    )
    assert matched["matched_rows"] == 2
