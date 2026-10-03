import json
from pathlib import Path

import pandas as pd
import pytest

from data_agent import config
from data_agent.datasets import DatasetError, resolve_dataset
from data_agent.ml import train_model
from data_agent.tools import train_ml_model


@pytest.fixture()
def data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    data = tmp_path / "data"
    data.mkdir()
    monkeypatch.setattr(config.settings, "data_dir", data.resolve())
    return data


def test_rejects_non_tabular_upload_types(data_dir: Path) -> None:
    (data_dir / "notes.json").write_text("[]", encoding="utf-8")
    (data_dir / "table.parquet").write_bytes(b"PAR1")
    with pytest.raises(DatasetError, match="csv"):
        resolve_dataset("notes.json")
    with pytest.raises(DatasetError, match="csv"):
        resolve_dataset("table.parquet")


def test_linear_regression_fits_a_line(data_dir: Path) -> None:
    frame = pd.DataFrame({"x": range(30), "y": [3 * i + 2 for i in range(30)]})
    frame.to_csv(data_dir / "line.csv", index=False)
    result = train_model("line.csv", target="y", method="linear_regression", features=["x"])
    assert result["task"] == "regression"
    assert result["metrics"]["r2"] > 0.99
    assert result["coefficients"]


def test_knn_classifies_two_groups(data_dir: Path) -> None:
    rows = [{"width": 1 if i < 20 else 10, "height": i % 3, "label": "small" if i < 20 else "large"} for i in range(40)]
    pd.DataFrame(rows).to_csv(data_dir / "groups.csv", index=False)
    result = train_model(
        "groups.csv",
        target="label",
        method="knn",
        features=["width", "height"],
        task="classification",
        n_neighbors=3,
    )
    assert result["task"] == "classification"
    assert result["metrics"]["accuracy"] >= 0.8


def test_svm_regression_tool_returns_json(data_dir: Path) -> None:
    frame = pd.DataFrame({"x": range(24), "y": [i * i for i in range(24)]})
    frame.to_csv(data_dir / "curve.csv", index=False)
    payload = json.loads(
        train_ml_model.invoke(
            {
                "name": "curve.csv",
                "target": "y",
                "method": "svm",
                "features_json": '["x"]',
                "task": "regression",
            }
        )
    )
    assert payload["method"] == "svm"
    assert "rmse" in payload["metrics"]
