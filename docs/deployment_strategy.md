# Deployment Strategy

## Overview

Three-phase deployment: POC demo → Pilot with real data → Production at NinjaVan scale.

---

## Phase 1 — POC / Demo (Now)

**Goal:** Working prototype for capstone evaluation.

| Component | Tool | Notes |
|-----------|------|-------|
| UI / Dashboard | FastAPI + custom HTML | Tailwind CSS + Plotly.js; served as static files |
| App Hosting | Hugging Face Spaces (Docker) | Dockerfile with python:3.11-slim + libgomp1; GitHub auto-deploy |
| Agent Backend | FastAPI (same container) | LangGraph + ML inference; no separate Modal needed for POC |
| Primary LLM | Gemini 2.5 Flash API | Google managed, no hosting needed |
| Fallback LLM | Groq Llama-3.3-70B | Activates only when Gemini returns 429/rate-limit; optional (GROQ_API_KEY) |
| Web Search | DuckDuckGo (ddgs) | No API key required; chatbot fallback layer |
| Vector DB | ChromaDB (embedded, 5 collections) | Rebuilt from .txt files at container startup if empty |
| Training | VSCode + conda (ninjavan env) | Models saved as .pkl to src/models/ and committed to GitHub |

**Deploy steps:**
1. Push code to GitHub
2. Link repo to Hugging Face Spaces → auto-deploys via Docker
3. Set secrets in HF Space Settings: `GEMINI_API_KEY`, optionally `GROQ_API_KEY`
4. ChromaDB auto-builds from `data/rag_documents/*.txt` on first startup

**Environment variables required:**
```
GEMINI_API_KEY=...       # Required — chatbot + decomposer
GROQ_API_KEY=...         # Optional — rate-limit fallback LLM; activates when Gemini returns 429 (free at console.groq.com)
```

---

## Phase 2 — Pilot (3–6 Months)

**Goal:** Test with real NinjaVan data in a single hub/region.

| Component | Tool | Notes |
|-----------|------|-------|
| UI | FastAPI + HTML | Same, add authentication layer |
| Hosting | Modal (scaled up) or GCP Cloud Run | Add persistent volume for ChromaDB |
| Monitoring | LangSmith | Track agent decisions, latency, errors per node |
| Data pipeline | Internal ops data → S3/GCS | Replace synthetic data with real parcel/sensor data |
| Model retraining | Scheduled cron / Modal | Monthly retraining trigger; retrain on new fraud/demand data |

**Success criteria:**
- Demand forecast MAPE < 15%
- Route optimization first-attempt rate > 85%
- Fraud detection precision > 80%
- Chatbot deflection rate > 40%
- Chatbot ChromaDB hit rate > 75% (minimal web fallback needed)

---

## Phase 3 — Production (12+ Months)

**Goal:** Enterprise-scale deployment across all NinjaVan markets.

| Component | Tool | Notes |
|-----------|------|-------|
| UI | FastAPI → internal ops portal | Integrate with NinjaVan NinjaDash |
| Hosting | GCP / AWS (Kubernetes) | High availability, multi-region |
| Agent Backend | GCP Cloud Run (containerized) | Auto-scaling; separate service per agent |
| Primary LLM | Gemini 2.5 Flash API (enterprise tier) | Higher rate limits, SLA guarantees |
| Fallback LLM | Groq or Vertex AI | Managed, enterprise SLA |
| Vector DB | Pinecone (managed, multi-tenant) | Production-grade, per-merchant collections |
| CI/CD | GitHub Actions | Automated test + deploy on merge |
| Monitoring | LangSmith + Datadog | Full observability: agent traces, latency, token cost |
| Data | NinjaVan internal data warehouse | Real-time streaming via Kafka |

---

## Architecture Diagram

```
[GitHub] ──push──> [Hugging Face Spaces / Docker]
                          │
                    FastAPI Backend
                          │
            ┌─────────────┼─────────────┐
            │             │             │
      LangGraph      ChromaDB      ML Models
    Control Tower   (5 collections)  (.pkl files)
            │
      ┌─────┴──────┐
      │  Chatbot   │──> Gemini 2.5 Flash (RAG)
      │  Pipeline  │──> DuckDuckGo Search
      │            │──> Groq Llama-3.3-70B (fallback)
      └────────────┘
```
