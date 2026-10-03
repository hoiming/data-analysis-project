from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from data_agent.agent import _turn_from_messages
from data_agent.memory import memory_session


def embed(texts: list[str]) -> list[list[float]]:
    vectors = []
    for text in texts:
        lowered = text.lower()
        vectors.append(
            [
                1.0 if "revenue" in lowered else 0.0,
                1.0 if "north" in lowered else 0.0,
                1.0 if "chart" in lowered else 0.0,
                float(len(lowered) % 7),
            ]
        )
    return vectors


def test_semantic_memory_round_trip(tmp_path: Path) -> None:
    with memory_session(tmp_path, embed, dims=4) as memory:
        memory.save_preference("Prefer bar charts.")
        memory.save_finding(
            "North revenue",
            "North has the highest revenue.",
            dataset="sample_sales.csv",
        )
        matches = memory.search("which region led revenue")
        texts = " ".join(item["value"]["text"] for item in matches)
        assert "highest revenue" in texts
        assert memory.prompt_block("revenue by region")
        memory.remember_thread("thread-1", "Revenue question")
        assert memory.list_threads()[0]["id"] == "thread-1"


def test_turn_collects_chart_from_tool_message(tmp_path: Path) -> None:
    chart = tmp_path / "chart.png"
    chart.write_bytes(b"png")
    from data_agent import config

    previous = config.settings.artifacts_dir
    config.settings.artifacts_dir = tmp_path
    try:
        turn = _turn_from_messages(
            [
                HumanMessage(content="Plot it"),
                AIMessage(
                    content="",
                    tool_calls=[{"id": "1", "name": "plot_dataset", "args": {}}],
                ),
                ToolMessage(
                    content='{"path": "%s", "chart_type": "bar"}' % chart,
                    name="plot_dataset",
                    tool_call_id="1",
                ),
                AIMessage(content="North leads."),
            ]
        )
    finally:
        config.settings.artifacts_dir = previous
    assert turn.charts == [str(chart)]
    assert "North leads." in turn.text
    assert "plot_dataset" in turn.tools
