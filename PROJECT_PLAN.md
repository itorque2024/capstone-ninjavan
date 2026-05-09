# NinjaVan AI Capstone — Project Plan
**Team Contact:** shaun@ninjavan.co
**Duration:** 7 Days
**Theme:** Operations Intelligence Suite

---

## Project Overview

Design and build an AI-powered Operations Intelligence Suite for a logistics company (NinjaVan-scale). The system uses 5 AI solutions orchestrated by a Multi-Agent Control Tower to improve demand planning, fleet health, fraud prevention, and customer service.

### Why This Angle
Previous capstone groups both built **Driver Copilot** apps (last-mile + route + driver chatbot). This project differentiates by targeting the **operations and business intelligence layer** — the brain behind the fleet, not the driver interface.

---

## Selected 5 Problems

| # | Problem | AI Approach | Key Differentiator |
|---|---------|-------------|-------------------|
| 1 | Demand Forecasting | ML + Deep Learning (LSTM + Prophet ensemble) | Covers ML + DL requirement |
| 2 | Intelligent Route Optimization | Optimization (TSP Nearest Neighbor) | Visual interactive mapping |
| 5 | Fraud Detection | ML (Isolation Forest + LightGBM) on Parcel Claims | Not done by previous groups |
| 6 | RAG Customer Chatbot | GenAI + RAG (Claude + ChromaDB) | Different from driver chatbots |
| 10 | Multi-Agent Control Tower | Agentic AI + Optimization (LangGraph) | Ties all 4 agents together |

### AI Types Coverage (All Required Types Covered)
- **ML** — Problems 1, 5
- **Deep Learning** — Problem 1 (LSTM)
- **GenAI** — Problem 6 (Claude API)
- **RAG** — Problem 6 (ChromaDB retrieval)
- **Agentic AI** — Problem 10 (LangGraph)
- **Optimization** — Problem 10 (LP optimizer inside Control Tower)

---

## Tech Stack

### Development
| Layer | Tool | Purpose |
|-------|------|---------|
| Language | Python 3.11 | All development |
| IDE | VSCode (local) | Code editing only — no local installs |
| Notebooks | Google Colab | Model training + GPU (notebooks install their own deps) |
| Version Control | GitHub (SSH) | Source of truth |
| Diagrams | draw.io | Architecture diagrams |

### AI / ML
| Layer | Tool | Purpose |
|-------|------|---------|
| Multi-Agent | LangGraph | Agent orchestration + state management |
| LLM | Claude API (Anthropic) | RAG chatbot + agent reasoning |
| ML Models | scikit-learn, Isolation Forest, LightGBM | Tabular ML (fraud detection) |
| Deep Learning | TensorFlow/Keras | LSTM demand forecasting |
| RAG Vector DB | ChromaDB | Document retrieval (embedded) |
| Forecasting | Prophet | Seasonal trend decomposition |
| Anomaly Detection | Isolation Forest | Fraud unsupervised detection |

### Hosting & Deployment
| Layer | Tool | Purpose |
|-------|------|---------|
| UI / Demo | Streamlit | Multi-tab dashboard + chatbot |
| App Hosting | Hugging Face Spaces | Free, GitHub auto-deploy, ML standard |
| Agent Backend | Modal | Serverless Python — LangGraph + ML inference |
| LLM API | Anthropic Claude API | Hosted, no self-hosting needed |

### Deployment Phases
```
Phase 1 (POC/Demo):   Streamlit → Hugging Face Spaces + Modal serverless backend
Phase 2 (Pilot):      Modal → scale up, add monitoring via LangSmith
Phase 3 (Production): Modal → containerized + GCP / AWS for enterprise scale
```

---

## Repository Structure

```
capstone-ninjavan/
├── PROJECT_PLAN.md
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
│
├── data/
│   ├── demand_forecasting/        # Kaggle: Brazilian E-Commerce or Instacart
│   ├── route_optimization/        # Real Singapore Postal Code geo-coordinates
│   ├── fraud_detection/           # Synthetic NinjaVan parcel claims dataset
│   └── rag_documents/             # NinjaVan FAQs, policies, SOPs (synthetic)
│
├── notebooks/                     # Run in Google Colab
│   ├── 01_demand_forecasting.ipynb
│   ├── 02_route_optimization.ipynb
│   ├── 03_fraud_detection.ipynb
│   └── 04_rag_chatbot.ipynb
│
├── src/
│   ├── agents/                    # LangGraph agents
│   │   ├── demand_agent.py
│   │   ├── maintenance_agent.py
│   │   ├── fraud_agent.py
│   │   ├── customer_agent.py
│   │   └── control_tower.py      # Orchestrator
│   ├── models/                    # Saved trained models
│   └── utils/
│       ├── data_loader.py
│       └── evaluation.py
│
├── app/
│   └── streamlit_app.py          # Main demo UI + dashboard
│
├── docs/
│   ├── architecture_diagram.png
│   ├── business_analysis.md
│   ├── model_justification.md
│   ├── deployment_strategy.md
│   ├── risk_and_ethics.md
│   └── roi_estimation.md
│
└── presentation/
    └── slide_outline.md
```

---

## Deliverables Checklist

| Deliverable | Owner | Status | Due |
|-------------|-------|--------|-----|
| Business problem analysis (5 problems) | All | Not started | Day 1 |
| AI architecture diagram | All | Not started | Day 1 |
| Dataset sourcing (Kaggle downloads) | All | Not started | Day 1 |
| Model justification document | All | Not started | Ongoing |
| Demand Forecasting notebook (LSTM + Prophet) | All | Not started | Day 2 |
| Route Optimization notebook (TSP Heuristic) | All | Not started | Day 3 |
| Fraud Detection notebook (Isolation Forest + LightGBM) | All | Not started | Day 3 |
| RAG Chatbot (Claude + ChromaDB) | All | Not started | Day 4 |
| LangGraph Multi-Agent Control Tower | All | Not started | Day 5 |
| Streamlit app + dashboard | All | Not started | Day 6 |
| Deployment strategy document | All | Not started | Day 6 |
| Risk & ethics document | All | Not started | Day 6 |
| ROI estimation document | All | Not started | Day 6 |
| Presentation slides | All | Not started | Day 7 |
| 10-minute video | All | Not started | Day 7 |

---

## 7-Day Sprint Plan

### Day 1 — Foundation
**Goal:** Repo live, data ready, architecture documented.

| Task | Details |
|------|---------|
| Set up GitHub repo | Create repo, clone locally in VSCode, connect to Colab |
| Create project structure | All folders + empty files per repo structure above |
| Download datasets | Kaggle: AI4I 2020, Brazilian E-Commerce, synthetic fraud data |
| Write business analysis | 1 paragraph per problem: pain point, cost, why AI |
| Draw architecture diagram | draw.io: all 5 components + LangGraph orchestration layer |
| EDA (exploratory data analysis) | Quick notebook per dataset: shape, nulls, distributions |

**End of Day 1:** Repo live on GitHub, 3 datasets downloaded, architecture diagram done.

---

### Day 2 — Demand Forecasting (Problem 1)
**Goal:** Working LSTM + Prophet model with evaluation metrics.

| Task | Details |
|------|---------|
| Data preprocessing | Parse dates, aggregate by region/day, handle missing values |
| Prophet baseline | Fit Prophet model, plot forecast with confidence intervals |
| LSTM model | Build sequence model in Keras, train on Colab GPU |
| Ensemble | Combine Prophet + LSTM predictions (weighted average) |
| Evaluation | MAE, RMSE, MAPE — compare vs naive baseline |
| Model justification | Write: why LSTM + Prophet over ARIMA/plain ML |

**End of Day 2:** Trained model saved, evaluation metrics logged, justification written.

---

### Day 3 — Route Optimization + Fraud Detection (Problems 2 & 5)
**Goal:** Routing engine and fraud classifier ready.

**Morning — Route Optimization:**

| Task | Details |
|------|---------|
| Geospatial mapping | Map delivery zones to lat/lon coordinates |
| TSP Algorithm | Implement Nearest Neighbor heuristic for route sequencing |
| Weather integration | Add real-time logic for rain delays |
| Visualization | Plot optimized route on Plotly map |

**Afternoon — Fraud Detection:**

| Task | Details |
|------|---------|
| Data preprocessing | Normalize claim amounts, engineer behavioral features |
| Isolation Forest | Unsupervised anomaly detection — find outlier claims |
| LightGBM | Supervised classifier on labeled fraud cases |
| Evaluation | Precision-recall curve, confusion matrix |
| Threshold tuning | Set decision threshold to minimize false negatives |

**End of Day 3:** 2 trained models saved, notebooks complete, metrics documented.

---

### Day 4 — RAG Customer Chatbot (Problem 6)
**Goal:** Working chatbot that retrieves answers from NinjaVan documents using Claude.

| Task | Details |
|------|---------|
| Create knowledge base | Write synthetic NinjaVan FAQs, SOP docs, policy text (20–30 documents) |
| Set up ChromaDB | Chunk documents, generate embeddings, store in local vector DB |
| Build RAG pipeline | Query → retrieve top-k docs → Claude prompt with context |
| Prompt engineering | System prompt: answer only from retrieved context, escalate if unsure |
| Test chatbot | Sample queries: tracking, failed delivery, rescheduling, complaints |
| Hallucination guard | Add confidence check — if retrieval score low, say "I don't know" |

**End of Day 4:** Chatbot answers from documents, tested on 10+ sample queries.

---

### Day 5 — Multi-Agent Control Tower (Problem 10)
**Goal:** LangGraph orchestrator that coordinates all 4 agents.

| Task | Details |
|------|---------|
| Design agent graph | Map: input → router → which agent → output → state update |
| Build Demand Agent | Calls demand forecasting model, returns forecast + alert if spike |
| Build Maintenance Agent | Calls predictive maintenance model, returns risk score per vehicle |
| Build Fraud Agent | Calls fraud model, returns flag + reason for suspicious claims |
| Build Customer Agent | Calls RAG chatbot, returns answer or escalation |
| Build Control Tower | LangGraph StateGraph orchestrating all 4 agents |
| Add Optimization layer | Simple LP: given forecast + fleet availability → recommend dispatch plan |
| Simulate scenario | Run end-to-end: "high demand period detected" → all agents respond |

**End of Day 5:** Control Tower runs a full multi-agent simulation.

---

### Day 6 — Streamlit App + Documentation
**Goal:** Working demo UI and all written deliverables complete.

**Morning — Streamlit App:**

| Task | Details |
|------|---------|
| Dashboard home | KPI cards: forecast accuracy, fleet health %, fraud rate, tickets deflected |
| Demand Forecasting tab | Interactive chart: historical + forecast + confidence bands |
| Intelligent Route Optimization tab | Interactive map visualization, distance and duration metrics |
| Fraud Detection tab | Claims flagged today, anomaly score distribution |
| RAG Chatbot tab | Chat interface for customer queries |
| Control Tower tab | Agent simulation: trigger a scenario, watch agents respond |

**Afternoon — Documentation:**

| Task | Details |
|------|---------|
| Deployment strategy | Phase 1 (HF Spaces + Modal) → Phase 2 (Modal scaled) → Phase 3 (GCP/AWS enterprise) |
| Risk & ethics | Table: 5 risks + mitigations, data privacy (PDPA), human-in-the-loop |
| ROI estimation | Per-problem savings estimate with industry benchmark sources |
| Model justification | Finalize all 5 justifications in docs/model_justification.md |

**End of Day 6:** Streamlit app demo-ready, all written docs complete.

---

### Day 7 — Presentation + Video
**Goal:** Slides done, video recorded, repo clean.

| Task | Details |
|------|---------|
| Build presentation slides | Follow 10-min structure below |
| Rehearse demo | Run Streamlit app live, practice Control Tower simulation |
| Record video | 10 minutes: problem → solution → live demo → impact |
| Final repo cleanup | README, .env.example, .gitignore, requirements.txt |
| Push everything to GitHub | Tag release v1.0 |

**10-Minute Video Structure:**

| Time | Slide / Content |
|------|----------------|
| 0:00–1:00 | Hook: logistics pain points, why AI now, project overview |
| 1:00–2:00 | Architecture diagram — how all 5 solutions connect |
| 2:00–4:30 | Walk through 5 problems (30s each: pain → solution → model choice) |
| 4:30–7:00 | Live Streamlit demo — dashboard + chatbot + agent simulation |
| 7:00–8:30 | Deployment strategy + ROI numbers |
| 8:30–9:30 | Risk & ethics |
| 9:30–10:00 | Summary: business impact, what we learned, next steps |

---

## Dataset Sources

| Problem | Dataset | URL |
|---------|---------|-----|
| Demand Forecasting | Brazilian E-Commerce (Olist) | kaggle.com/datasets/olistbr/brazilian-ecommerce |
| Intelligent Route Optimization | Singapore Postal Districts (OpenStreetMap) | Nominatim API / Local Dictionary |
| Fraud Detection | NinjaVan Parcel Claim Fraud (Synthetic) | Generated internally based on Ops profiles |
| RAG Chatbot | Synthetic (self-created FAQ docs) | Created in Day 4 |
| Control Tower | Simulated from above datasets | Generated in Day 5 |

---

## Model Justification Summary

| Problem | Chosen Model | Key Reason |
|---------|-------------|-----------|
| Demand Forecasting | LSTM + Prophet ensemble | LSTM for non-linear patterns, Prophet for seasonality — ensemble beats either alone |
| Intelligent Route Optimization | TSP Nearest Neighbor | Computationally efficient for dynamic daily dispatching, easy to integrate with weather heuristics |
| Fraud Detection | Isolation Forest + LightGBM | Isolation Forest catches unknown fraud patterns; LightGBM handles labeled cases |
| RAG Chatbot | Claude + ChromaDB | RAG avoids retraining cost; Claude best-in-class for reasoning; ChromaDB lightweight |
| Control Tower | LangGraph + LP Optimizer | Stateful graph execution, built-in tool-calling, explicitly recommended in brief |

---

## Risk & Ethics Summary

| Risk | Impact | Mitigation |
|------|--------|-----------|
| Biased fraud model flags legitimate customers | High | Fairness audit, human review before action |
| Route optimizer suggests illegal/unsafe roads | Medium | Restrict paths to commercial routing APIs in production |
| LSTM fails on black swan events | Medium | Uncertainty bounds, rule-based fallback |
| RAG chatbot hallucination | Medium | Answer only from retrieved docs, confidence threshold |
| Over-reliance on agent decisions | High | Agents recommend; humans approve high-stakes actions |

---

## ROI Estimation

| Problem | Annual Saving Estimate | Basis |
|---------|----------------------|-------|
| Demand Forecasting | 8–12% fleet cost reduction | Gartner logistics ML benchmark |
| Intelligent Route Optimization | 10–15% fuel cost reduction | Industry standard for TSP heuristics vs manual planning |
| Fraud Detection | 20–30% reduction in false claims | Industry avg fraud = 2–5% of revenue |
| RAG Chatbot | 40% CS ticket deflection | Standard RAG chatbot industry benchmark |
| Control Tower | 15% ops efficiency gain | Multi-agent coordination studies |

---

## Evaluation Rubric Coverage

| Criteria | Weight | How We Cover It |
|----------|--------|----------------|
| Problem Understanding | 20% | Business analysis doc + clear pain points per problem |
| AI Design & Justification | 30% | Model justification doc + architecture diagram |
| Technical Feasibility | 20% | Working Streamlit prototype + trained models |
| Business Impact | 15% | ROI estimation + industry benchmark numbers |
| Innovation (Agentic/GenAI) | 15% | LangGraph Control Tower + Claude RAG chatbot |
