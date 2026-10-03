# Data analysis agent

A local analyst you talk to in a Gradio chat. It uses LangChain tool calling on a LangGraph agent, DeepSeek through the OpenAI-compatible API, and two kinds of memory:

- **Conversation memory** — a SQLite checkpoint per chat thread, so a thread can be resumed.
- **Long-term memory** — preferences and saved findings in SQLite, searched by meaning.

DeepSeek does not offer an embeddings API. Semantic search uses a local FastEmbed model (`BAAI/bge-small-en-v1.5`), downloaded on the first memory read or write.

## Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
```

Put your key in `.env`:

```bash
DEEPSEEK_API_KEY=sk-...
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat
```

`deepseek-chat` is the tool-calling model. Then:

```bash
python -m data_agent.app
```

Open the local URL Gradio prints. Upload a CSV or Excel file in the sidebar, or put a `.csv`, `.xlsx`, or `.xls` file in `data/` yourself. `data/sample_sales.csv` is included.

## What the agent can do

It can list files, describe columns, filter rows, aggregate (including `units * price` via `sum_product`), count values, correlate numeric columns, and save a matplotlib chart into `artifacts/`. The chat shows that chart. It can also fit scikit-learn models: linear regression, SVM, and KNN, and report held-out scores.

It cannot run arbitrary Python. Filters and aggregations are structured tool arguments.

## Memory files

- `memory/checkpoints.sqlite` — messages for each conversation
- `memory/store.sqlite` — preferences, findings, and their vectors
