# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A 7-day capstone project building an **Operations Intelligence Suite** for NinjaVan (logistics). Five AI solutions are orchestrated through a LangGraph multi-agent control tower. Covers all 6 required AI types: ML, Deep Learning, GenAI, RAG, Agentic AI, and Optimization (scipy LP).

Selected problems: Demand Forecasting (1), Predictive Maintenance (4), Fraud Detection (5), RAG Chatbot (6), Multi-Agent Control Tower with LP Optimizer (10).

## Conda Environment (Mac M2)

Lewis uses **conda** to manage all Python environments. Never install packages into base or system Python.

```bash
# One-time setup
conda env create -f environment.yml
conda activate ninjavan

# Register kernel so VSCode notebooks can use it
python -m ipykernel install --user --name ninjavan --display-name "Python (ninjavan)"

# Install pre-commit security hooks
pre-commit install

# Run the FastAPI app locally
uvicorn app.main_api:app --reload
# Open http://127.0.0.1:8000

# Deploy to Hugging Face Spaces
git checkout hf-sync
git checkout main -- <changed files>
PRE_COMMIT_ALLOW_NO_CONFIG=1 git commit -m "..."
git push space hf-sync:main
git checkout main
```

Select the **ninjavan** kernel in VSCode when opening any notebook (bottom-right kernel picker).

## Environment Variables

Copy `.env.example` to `.env` at the repo root:
```
GEMINI_API_KEY=AIza...   # Required — RAG chatbot + customer agent (free tier at aistudio.google.com)
LANGSMITH_API_KEY=...    # Optional — LangGraph tracing
```

**Never commit `.env`.** For Hugging Face Spaces: set `GEMINI_API_KEY` in Space Settings → Secrets.

## Architecture

### Multi-Agent Control Tower (`src/agents/control_tower.py`)

LangGraph graph node sequence for every event:

```
START → [router on event_type] → demand_agent | maintenance_agent | fraud_agent | customer_agent
                                                                                        ↓
                                                                               aggregate_alerts
                                                                                        ↓
                                                                               fleet_optimizer  ← scipy LP
                                                                                        ↓
                                                                                       END
```

`TowerState` carries all inputs and outputs as a single `TypedDict`. The `fleet_optimizer` node always runs after aggregation — it uses `scipy.optimize.linprog` to allocate available (non-flagged) fleet vehicles across the forecast horizon, writing to `dispatch_plan`.

| `event_type` | Agent | Model file |
|---|---|---|
| `"demand"` | `demand_agent.py` | `src/models/demand_model.pkl` |
| `"maintenance"` | `maintenance_agent.py` | `src/models/maintenance_model.pkl` |
| `"fraud"` | `fraud_agent.py` | `src/models/fraud_model.pkl` |
| `"customer"` | `customer_agent.py` | ChromaDB + Claude API (no pkl) |

### Model Wrappers (defined in notebooks, saved as pkl)

The pkl files are **custom wrapper classes**, not raw sklearn models:

- `DemandForecastModel` — `predict(horizon: int) → {"baseline_avg": float, "values": [float, ...]}`
- `MaintenanceRiskModel` — `predict_proba(df) → array[:, 1]` — imputes missing sensor columns from `vehicle_health_score` alone
- `FraudDetectionModel` — `decision_function(df) → array` — imputes features from `parcel_value` alone

### Chatbot Orchestrator (`src/agents/chatbot/orchestrator.py`)

Three-node LangGraph pipeline: **Decomposer → Processor → Synthesizer**

- **Decomposer**: Gemini splits the customer message into N sub-questions, each tagged with an intent (`tracking`, `delivery`, `claims`, `policy`, `ops`, `escalation`).
- **Processor**: For each sub-question, queries the matching ChromaDB collection. If docs found → Gemini synthesises a RAG answer. If no docs → DuckDuckGo web search → Gemini (or Groq on 429) answers from web context.
- **Synthesizer**: Single sub-question returns directly; multiple sub-questions are merged with per-agent section headers.

Web fallback uses `_gemini()` (Groq only activates on 429, not by default). Prompts include `"Never mention internal document names, filenames, or file paths."` to prevent leakage.

LLM source badge on frontend: `"RAG + Gemini"` when ChromaDB docs were retrieved; `"via Gemini"` for web/direct answers.

### RAG Knowledge Base (`src/agents/chatbot/_llm.py`)

`_MODEL = "gemini-2.5-flash"` (google-genai SDK). `genai.Client` is a module-level singleton. `get_last_llm()` tracks which LLM answered last for the source badge.

### ChromaDB (`src/utils/chroma_setup.py`)

- `_DEFAULT_CHROMA_PATH` — absolute path anchored to repo root, used by both `orchestrator.py` and `main_api.py`
- `build_chroma_from_files()` — called at FastAPI startup when `collection.count() == 0`; reads all `.txt` files from `data/rag_documents/`
- `data/rag_documents/*.txt` (**35 files**) — **committed to git**; `chroma_db/` is gitignored and rebuilt at runtime
- All 35 files verified against ninjavan.co: email `shippercare_sg@ninjavan.co`, SLA 1–3 working days, hours Mon–Sat 9AM–8PM SGT, international rates (ID from S$8, HK from S$10)

### Training Notebooks (`notebooks/`)

Run in VSCode with the `ninjavan` kernel. Each notebook saves its pkl to `src/models/`, then commit and push manually.

| Notebook | Output | Agent |
|---|---|---|
| `01_demand_forecasting.ipynb` | `demand_model.pkl`, `demand_scaler.pkl` | `demand_agent.py` |
| `02_predictive_maintenance.ipynb` | `maintenance_model.pkl` | `maintenance_agent.py` |
| `03_fraud_detection.ipynb` | `fraud_model.pkl` | `fraud_agent.py` |
| `04_rag_chatbot.ipynb` | Tests RAG pipeline only (no pkl) | `customer_agent.py` |

### FastAPI Backend + Frontend (`app/main_api.py`, `app/static/index.html`)

Five API endpoints, each powering a dashboard view:

| Endpoint | View | Notes |
|---|---|---|
| `POST /api/demand` | Volume Forecast | Returns forecast chart JSON + insights |
| `POST /api/route` | Route Planner | Returns Scattermapbox JSON + metrics |
| `POST /api/fraud` | Fraud Scanner | Returns queue, `claims_data` (slim, for JS filtering), `plot_json` |
| `POST /api/simulate` | Command Centre | Runs full 5-agent control tower |
| `POST /api/chat` | Customer Support | Runs chatbot orchestrator pipeline |

**Fraud Scanner specifics:**
- `_generate_fraud_sample(500)` — runtime synthetic fallback when `ninjavan_optionB_datasets/fraud_data.csv` is absent (e.g. HF Spaces). Uses `RandomState(42)` for reproducibility.
- `claims_data` in API response: slim per-claim list `{id, age, claims, risk}` enabling client-side filtering without re-scanning.
- Frontend `filterFraudChart(claimId)`: if `claimId` is set → shows 1 highlighted dot; else → shows all claims ≥ current threshold. Both threshold field (`onchange`) and claim select (`onchange`) trigger this function.
- Chart uses `go.Scatter` with explicit `.tolist()` arrays (avoids px.scatter colour-axis trace quirks that caused empty charts).

## What Is and Is Not Tracked by Git

| Path | Tracked? | Notes |
|---|---|---|
| `src/models/*.pkl` | **Yes** | Commit after each notebook run; how models reach HF Spaces |
| `src/models/*.keras`, `*.h5` | No | TF models are large; pkl wrappers use Prophet-only inference |
| `chroma_db/` | No | Rebuilt from txt files at runtime |
| `data/rag_documents/*.txt` | **Yes** | Source of truth for ChromaDB |
| `ninjavan_optionB_datasets/*.csv` | **Yes** | Training datasets |
| `data/**/*.csv` | No | Would be processed/output data |
| `.env` | No | Never commit; use HF Secrets for deployment |

## Security

- `pre-commit install` activates hooks defined in `.pre-commit-config.yaml`:
  - Blocks commits containing `AIza...` (Gemini key pattern)
  - Blocks committing `.env` directly
  - Runs `detect-private-key` on all staged files
- All API keys read via `os.environ` / `load_dotenv()` only — never hardcoded

## Deployment (Hugging Face Spaces)

The `README.md` top has the required HF Spaces YAML block (`sdk: streamlit`, `app_file: app/streamlit_app.py`). On push to GitHub, HF Spaces auto-deploys. Set `GEMINI_API_KEY` in Space Settings → Secrets before the first deployment.

## HF Spaces Deployment Pattern

The `hf-sync` branch tracks `space/main` (Hugging Face). Main branch stays clean.

```bash
git checkout hf-sync
git checkout main -- <file1> <file2>   # cherry-pick specific files from main
# exit code 2 = LFS warning only, not a failure — check git diff --cached to confirm
PRE_COMMIT_ALLOW_NO_CONFIG=1 git commit -m "..."
git push space hf-sync:main
git checkout main
```

Never commit `.env`, `chroma_db/`, `notebooks/`, or `__pycache__/` to hf-sync.

## Key Technical Notes

- **Keras 3.x**: LSTM uses `model()` call syntax, `.keras` format, loss as class instance (`tf.keras.losses.MeanSquaredError()`).
- **LangGraph 0.2.50**: Agent nodes are plain functions. `add_conditional_edges(START, router, {...})` for routing.
- **ChromaDB path**: Always use `_DEFAULT_CHROMA_PATH` from `chroma_setup.py`. Never use `"./chroma_db"`.
- **Gemini model**: `_MODEL = "gemini-2.5-flash"` in `src/agents/chatbot/_llm.py`. Free tier is 15 RPM — Groq fallback activates on 429/RESOURCE_EXHAUSTED only.
- **Fraud chart**: Use `go.Scatter` with `.tolist()` on all arrays. `px.scatter` with `color` continuous scale can produce empty traces in some Plotly.js versions.
- **Fraud synthetic data**: `_generate_fraud_sample(500)` in `main_api.py` — `RandomState(42)` seed, no CSV dependency. Used on HF Spaces where the training CSV is not committed.
- **FastAPI app**: Entry point is `app/main_api.py`. Run with `uvicorn app.main_api:app --reload` locally. Dockerfile uses `CMD ["uvicorn", "app.main_api:app", "--host", "0.0.0.0", "--port", "7860"]`.
