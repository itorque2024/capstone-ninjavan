# Model Justification

## Problem 1 — Demand Forecasting: LSTM + Prophet Ensemble

| Model | Why Chosen | Why Not Alternatives |
|-------|-----------|----------------------|
| LSTM | Captures long-range non-linear temporal dependencies in shipment volume | RNN: vanishing gradient; plain FFNN: no temporal memory |
| Prophet | Handles daily/weekly/annual seasonality + holiday effects automatically | ARIMA: assumes linearity, poor on irregular seasonality |
| Ensemble | Combines LSTM's pattern recognition with Prophet's seasonality — reduces variance | Either alone has blind spots the other covers |
| Exogenous Features | Marketing Spend and Sale Calendars (11.11) injected into LSTM to predict massive spikes | Univariate models fail to predict marketing-driven demand |
| Granularity | Output mathematically subdivided into Warehouse-Hub level shares | "All of Singapore" forecast isn't actionable for local ops managers |

**Evaluation Metric:** MAPE (Mean Absolute Percentage Error) — interpretable as % forecast error for business stakeholders.

---

## Problem 2 — Intelligent Routing: TSP Nearest-Neighbour + Haversine Distance Matrix

| Model | Why Chosen | Why Not Alternatives |
|-------|-----------|----------------------|
| Haversine Distance Matrix | Calculates real geographic distances across Singapore's 28 postal districts without requiring a map API | Google Maps API: adds cost and latency; straight-line Euclidean: inaccurate on curved geography |
| TSP Nearest-Neighbour Heuristic | Calculates fastest delivery sequence instantly for real-time dispatch | Deep Reinforcement Learning: too slow for real-time dispatch at scale; Exact Solvers (Gurobi): computationally explosive for large fleets |
| Live Weather Factor | Pulls real rain data from Open-Meteo API; multiplies estimated travel time by `1 + rain_mm × 0.015` (1.5% per mm) | Static routing ignores weather; drivers miss SLAs without delay adjustment |
| Dynamic Rider Count | `ceil(parcels / 80)` riders — scales automatically with demand forecast | Fixed fleet size: over-deploys in low demand, under-deploys in spikes |

**Evaluation Metric:** Total Distance (km) + Weather Delay Factor + Decision Latency (ms) — proving the model is fast enough for dynamic dispatch.

---

## Problem 5 — Fraud Detection: Isolation Forest + LightGBM

| Model | Why Chosen | Why Not Alternatives |
|-------|-----------|----------------------|
| Isolation Forest | Unsupervised — detects anomalies instantly without needing labeled fraud data | Autoencoder: more complex, needs heavy tuning |
| LightGBM | Supervised — leverages labeled fraud history for high-precision classification | XGBoost: marginally slower; Logistic Regression: too linear for complex fraud patterns |
| Straight-Through Processing (STP) | Auto-approves 90%+ of normal claims, only routing high-risk anomalies to humans | Manual review: unscalable during mega sales |
| Dynamic Thresholding | AI dynamically relaxes strictness during 11.11 spikes (knowing legitimate damage rises) | Static rules: flag too many false positives during sales |

**Evaluation Metric:** Precision-Recall AUC — standard for highly imbalanced fraud detection problems.

---

## Problem 6 — Agentic RAG Chatbot: LangGraph + Gemini 2.5 Flash + ChromaDB + Groq

| Component | Why Chosen | Why Not Alternatives |
|-----------|-----------|----------------------|
| LangGraph StateGraph | Enables a stateful multi-node pipeline (decomposer → processor → synthesizer) with shared state | Simple function chain: no state passing; LangChain agents: less control over explicit routing |
| Gemini 2.5 Flash | Fast, cost-effective, strong reasoning; used for decomposition and RAG answering | GPT-4/Claude: too expensive for high-volume support; smaller models: weaker instruction following |
| Query Decomposer | Splits multi-question messages into N sub-questions, each tagged with the correct intent | Single-pass routing: one question type wins, rest ignored |
| 5 Specialist ChromaDB Collections | Each agent queries only its relevant domain (nv_tracking, nv_delivery, nv_claims, nv_policy, nv_general) | Single flat collection: retrieval pollution across unrelated topics; Pinecone: adds cost |
| RAG over Fine-tuning | No retraining cost; knowledge base updates instantly by adding .txt files; grounded answers | Fine-tuning: expensive, requires retraining on every policy change |
| DuckDuckGo Web Search | Free, no API key; searches the web when ChromaDB has no matching documents | Bing/Google Search API: paid; no fallback: "I don't know" dead-ends frustrate customers |
| Groq Llama-3.3-70B | Rate-limit fallback only — activates automatically when Gemini returns 429/RESOURCE_EXHAUSTED | Self-hosted LLM: high infra cost; using Groq as primary: less accurate than Gemini 2.5 Flash |
| Escalation Agent | Structured empathy response + human handoff for frustrated customers — no RAG needed | Generic fallback message: inflames already frustrated customers |

**Evaluation:** Manual QA on 25 test queries across all 6 intent types; escalation rate as proxy for chatbot confidence; source attribution (chromadb+gemini / web+gemini / web+groq [rate-limit only]) logged in debug trace per query.

---

## Problem 10 — Multi-Agent Control Tower: LangGraph

| Component | Why Chosen | Why Not Alternatives |
|-----------|-----------|----------------------|
| LangGraph StateGraph | Stateful agent graphs — 5 nodes share a single `TowerState` TypedDict; each node reads previous outputs and writes its own | n8n: no-code but less flexible for custom ML; AutoGen: less control over explicit state transitions |
| Sequential Pipeline | Demand → Route → Warehouse → Pricing → Customer → Coordinator; each agent's decision is directly informed by upstream output | Parallel agents: lose the cascading dependency (rider count must be known before warehouse slotting) |
| Global Demand Volume | `global_demand_volume` (peak forecast) passed via API to Fraud Scanner and RAG Chatbot — both adjust their behaviour automatically | Siloed operations: fraud threshold and chatbot messaging stay static during demand spikes |
| Coordinator Node | Reviews all agent decisions, detects conflicts (e.g., high surge pricing + long delivery times), calculates decision latency | No arbitration: conflicting agent decisions go unresolved |
| Scenario Simulation | Pre-built scenarios (11.11 spike / normal / custom multiplier) trigger the full 5-agent pipeline | Live data only: can't demonstrate peak-period coordination without a synthetic spike |

**Evaluation:** End-to-end simulation — visually demonstrates the 5 LangGraph agents coordinating in <0.05 seconds, with per-agent reasoning logs visible in the dashboard.
