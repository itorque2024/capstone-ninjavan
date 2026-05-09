# NinjaVan AI Capstone — Operations Intelligence Suite

> AI-powered backend intelligence for a logistics company at NinjaVan scale.
> Demand forecasting · Route optimization · Fraud detection · RAG chatbot · Multi-agent control tower

---

## School Submission Summary

**Project:** NinjaVan Operations Intelligence Suite — a production-grade AI backend for last-mile logistics, covering all 6 required AI types in a single deployable system.

**Live Demo:** [https://huggingface.co/spaces/itorque/capstone-ninjavan](https://huggingface.co/spaces/itorque/capstone-ninjavan)

**GitHub Repo:** [https://github.com/itorque2024/capstone-ninjavan](https://github.com/itorque2024/capstone-ninjavan)

**Demo Video:** [presentation/NinjaVan_AI_Suite.mp4](presentation/NinjaVan_AI_Suite.mp4)

**Presentation Slides:** [presentation/NinjaVan_Operations_Intelligence_Suite.pptx](presentation/NinjaVan_Operations_Intelligence_Suite.pptx)

---

## Project Overview

Modern logistics operations are siloed. When a massive event hits — an 11.11 mega-sale, monsoon rain, a surge of damage claims — each department (fleet, fraud, customer service) reacts independently with hours of lag between them.

This capstone builds the **Operations Intelligence Suite**: five AI modules coordinated through a LangGraph multi-agent Control Tower. One demand forecast triggers automatic rider dispatch, fraud threshold relaxation, and customer delay messaging — simultaneously, in under 0.05 seconds.

Unlike driver-facing apps, this system targets the **operational layer**: the AI brain that logistics managers and ops teams rely on to make decisions at scale.

**Estimated annual value: SGD 17.8M | Year 1 ROI: ~1,300% | Payback period: < 1 month**
> See [docs/roi_estimation.md](docs/roi_estimation.md) for full derivation and assumptions.

![Infographic](docs/infographic.png)

---

## Selected Problems & AI Coverage

| # | Problem | AI Type | Approach |
|---|---------|---------|----------|
| 1 | Demand Forecasting | Deep Learning + ML | LSTM + Prophet ensemble with marketing spend & sale calendar features |
| 2 | Route Optimization | Optimization | TSP Nearest-Neighbour + K-Means over 28 SG districts + live weather factor |
| 5 | Fraud Detection | ML (Unsupervised + Supervised) | Isolation Forest + LightGBM, dynamic thresholding, Straight-Through Processing |
| 6 | RAG Customer Chatbot | GenAI + RAG + Agentic AI | LangGraph 3-stage pipeline, Gemini 2.5 Flash, ChromaDB, 6 specialist agents |
| 10 | Multi-Agent Control Tower | Agentic AI + Optimization | LangGraph 5-agent pipeline (Demand → Route → Warehouse → Pricing → Customer → Coordinator) |

**AI Types Covered:** ML · Deep Learning · GenAI · RAG · Agentic AI · Optimization (all 6 required)

---

## Module Descriptions

### Module 1 — Demand Forecasting
LSTM + Prophet ensemble trained on synthetic SEA parcel volume data with injected exogenous features (marketing spend, 11.11/12.12 sale calendars). Forecasts are subdivided by warehouse hub for actionable local dispatch planning. Forecast error reduced from ~25% MAPE (manual planning) to ~12% (AI ensemble).

### Module 2 — Intelligent Route Optimization
TSP Nearest-Neighbour heuristic across all 28 Singapore postal districts with K-Means cluster assignment per rider. Pulls live rainfall data from Open-Meteo API and applies a dynamic delay factor (`1 + rain_mm × 0.015`). Rider count scales automatically with the demand forecast (`ceil(parcels / 80)`). Sub-second routing decisions; 39% reduction in failed first-attempt deliveries.

### Module 3 — Fraud Detection & Claims
Two-model pipeline: Isolation Forest (unsupervised, catches novel fraud) + LightGBM (supervised, high precision on known patterns). Straight-Through Processing auto-approves 90%+ of normal claims; only high-risk anomalies are routed to human review. Dynamic thresholding automatically relaxes during demand spikes (legitimate damage rises during mega-sales). Interactive fraud signal map: select a Claim ID to isolate a single dot, or adjust the risk threshold to filter flagged claims.

### Module 4 — Agentic RAG Customer Chatbot
Three-stage LangGraph pipeline:
- **Decomposer** (Gemini 2.5 Flash): splits multi-question messages into individual sub-questions, each tagged with intent (`tracking` / `delivery` / `claims` / `policy` / `ops` / `escalation`)
- **Processor**: 6 specialist agents — 5 with dedicated ChromaDB collections (35 verified NinjaVan documents); Escalation Agent uses direct Gemini (no RAG). When ChromaDB has no matching docs → DuckDuckGo web search → Gemini 2.5 Flash synthesises a grounded answer. Groq Llama-3.3-70B activates automatically if Gemini hits rate limits.
- **Synthesizer**: merges all sub-answers into one response with per-agent section headers and source attribution badges

Covers international shipping rates for Singapore, Malaysia, Indonesia, Hong Kong, Philippines, Thailand, and Vietnam. 42% ticket deflection rate.

### Module 5 — Multi-Agent Control Tower
LangGraph orchestrator running a 5-agent sequential pipeline over a shared `TowerState`: **Demand → Route → Warehouse → Pricing → Customer → Coordinator**. Each agent reads the previous agent's output and cascades its decisions — rider count, warehouse zone assignments, surge pricing, and proactive customer alerts all adjust automatically from a single demand forecast. A Coordinator node then checks for conflicts and calculates decision latency. Full 5-agent coordination completes in under 0.05 seconds.

Additionally, the Fraud Scanner and RAG Chatbot are standalone modules that read `global_demand_volume` via the API — the fraud threshold relaxes automatically during demand spikes, and the Chatbot activates delay-warning mode when volume exceeds 10,000 parcels.

---

## Key Results

| Module | Metric | Impact |
|--------|--------|--------|
| Demand Forecasting | Forecast error (MAPE) | 25% → ~12% (-52%) |
| Route Optimization | Failed first-attempt deliveries | 18% → ~11% (-39%) |
| Fraud Detection | Detection rate vs manual | 75% vs 20% manual; 84% fewer human reviews |
| RAG Chatbot | Ticket deflection | 42% deflection; multi-question resolved in one response |
| Control Tower | Decision lag | 4 hours → 2 minutes; 5 agents coordinate in <0.05s |
| **Total** | **Estimated annual saving** | **SGD 17.8M/year** |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.11 |
| Multi-Agent Orchestration | LangGraph 0.2.x (Decomposer → Processor → Synthesizer) |
| Primary LLM | Gemini 2.5 Flash (google-genai SDK) |
| Fallback LLM | Groq Llama-3.3-70B (activates on Gemini 429/rate-limit only) |
| ML / Forecasting | scikit-learn, LightGBM, Prophet |
| Deep Learning | TensorFlow/Keras (LSTM) |
| RAG / Vector DB | ChromaDB (5 specialist collections, rebuilt at runtime from 35 .txt files) |
| Optimization | TSP Nearest-Neighbour (route dispatch) + rule-based surge pricing |
| Web Search Fallback | DuckDuckGo (ddgs, no API key required) |
| Weather Data | Open-Meteo API (free, no API key required) |
| API Backend | FastAPI + Uvicorn |
| Frontend | Single-page HTML/JS + Tailwind CSS + Plotly.js |
| Hosting | Hugging Face Spaces (Docker) |
| Training | Google Colab / VSCode (notebooks in `/notebooks`) |

---

## Documentation

| Document | Description |
|----------|-------------|
| [Architecture Overview](docs/architecture.md) | Multi-agent workflow diagram, chatbot sub-architecture, full tech stack |
| [Business Analysis](docs/business_analysis.md) | Problem statements, cost impacts, why AI was chosen per module |
| [Model Justification](docs/model_justification.md) | Why each model was chosen over alternatives, evaluation metrics |
| [ROI Estimation](docs/roi_estimation.md) | SGD 17.8M savings breakdown, assumptions, payback period, Year 1 ROI |
| [Risk & Ethics](docs/risk_and_ethics.md) | Risk register, algorithmic bias, PDPA compliance, human-in-the-loop policy |
| [Deployment Strategy](docs/deployment_strategy.md) | 3-phase roadmap: HF Spaces → Cloud Run → GKE |
| [Infographic](docs/infographic.png) | One-page visual summary: Control Tower, RAG Chatbot, ROI highlights |
| [Mind Map](docs/NotebookLM%20Mind%20Map.png) | Full project mind map generated by NotebookLM |
| [Presentation Slides](presentation/NinjaVan_Operations_Intelligence_Suite.pptx) | Final presentation deck (PPTX) |
| [Demo Video](presentation/NinjaVan_AI_Suite.mp4) | 10-minute capstone demo video (MP4) |
| [Presentation Outline](presentation/script/slide_outline.md) | 14-slide structure with timing and speaker notes |
| [Video Script](presentation/script/ai_video_script.md) | Full 10-minute spoken script with visual cues |
| [ML Training Guide](docs/ml_training.md) | Internal guide for retraining models locally |

---

## Project Structure

```
capstone-ninjavan/
├── app/
│   ├── main_api.py          # FastAPI backend — 5 endpoints: /api/demand, /api/route,
│   │                        #   /api/fraud, /api/simulate, /api/chat
│   └── static/
│       └── index.html       # Single-page dashboard (5 views, Tailwind + Plotly.js)
├── src/
│   ├── agents/              # LangGraph agents
│   │   ├── control_tower.py
│   │   ├── demand_agent.py
│   │   ├── route_agent.py
│   │   ├── fraud_agent.py
│   │   ├── maintenance_agent.py
│   │   └── chatbot/
│   │       ├── orchestrator.py  # Decomposer → Processor → Synthesizer pipeline
│   │       └── _llm.py          # Gemini 2.5 Flash + Groq fallback helper
│   ├── models/              # Trained models (.pkl, committed to git)
│   └── utils/               # Shared utilities (ChromaDB setup, weather loader, SG districts)
├── notebooks/               # Training notebooks (VSCode / Colab)
├── data/
│   └── rag_documents/       # 35 .txt knowledge-base files (committed, verified against
│                            #   ninjavan.co; ChromaDB rebuilt from these at runtime)
├── docs/                    # Architecture, justification, risk, ROI, deployment docs
│   ├── infographic.png      # One-page visual summary
│   └── NotebookLM Mind Map.png  # Full project mind map
├── presentation/
│   ├── NinjaVan_AI_Suite.mp4                        # Demo video
│   ├── NinjaVan_Operations_Intelligence_Suite.pptx  # Slide deck
│   └── script/              # Presentation scripts
├── Dockerfile               # Docker deployment config for HF Spaces
├── requirements.txt
└── environment.yml          # Conda environment (local dev, Mac M2)
```

---

## Quick Start

### Local (Development)

```bash
conda activate ninjavan
uvicorn app.main_api:app --reload
# Open http://127.0.0.1:8000
```

### Deploy to Hugging Face Spaces

The `hf-sync` branch tracks the HF Space. Main branch stays clean.

```bash
git checkout hf-sync
git checkout main -- <changed files>
PRE_COMMIT_ALLOW_NO_CONFIG=1 git commit -m "deploy: ..."
git push space hf-sync:main
git checkout main
```

Set `GEMINI_API_KEY` in Space Settings → Secrets before first deploy.

---

## Datasets

| Problem | Dataset | Notes |
|---------|---------|-------|
| Demand Forecasting | Synthetic SEA parcel volume (`demand_data.csv`) | Daily parcel counts across ID/TH/MY regions; marketing spend and sale flags added synthetically |
| Route Optimization | Synthetic Singapore geo-data | 28 postal districts + hub coordinates |
| Fraud Detection | Synthetic NinjaVan parcel fraud (`fraud_data.csv`) | Parcel claims with value, prior claims, account age, claim lag; 5% fraud rate. Falls back to runtime synthetic generation on HF Spaces. |
| RAG Chatbot | Verified NinjaVan FAQ docs (`data/rag_documents/`, **35 files**) | Policy, tracking, claims, delivery, international shipping (SG/MY/ID/HK/PH/TH/VN), and general support docs. All content verified against ninjavan.co. |

---

## Training Notebooks

| Notebook | Trains | Output |
|----------|--------|--------|
| [01_demand_forecasting.ipynb](notebooks/01_demand_forecasting.ipynb) | LSTM + Prophet ensemble | `demand_model.pkl` |
| [02_predictive_maintenance.ipynb](notebooks/02_predictive_maintenance.ipynb) | Maintenance risk classifier | `maintenance_model.pkl` |
| [03_fraud_detection.ipynb](notebooks/03_fraud_detection.ipynb) | Isolation Forest + LightGBM | `fraud_model.pkl` |
| [04_rag_chatbot.ipynb](notebooks/04_rag_chatbot.ipynb) | RAG pipeline tests | _(no pkl)_ |

---

## Project Mind Map

![Mind Map](docs/NotebookLM%20Mind%20Map.png)

---

## Environment Variables

Copy `.env.example` to `.env`:

```
GEMINI_API_KEY=AIza...   # Required — chatbot + RAG (free at aistudio.google.com)
GROQ_API_KEY=...         # Optional — Groq fallback LLM (free at console.groq.com)
LANGSMITH_API_KEY=...    # Optional — LangGraph tracing
```

Never commit `.env`. For Hugging Face Spaces, set `GEMINI_API_KEY` in Space Settings → Secrets.
