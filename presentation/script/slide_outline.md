# 10-Minute Presentation — Slide Outline

## Slide 1 — Title (0:00–0:20)
- Project name: NinjaVan Operations Intelligence Suite
- Team members
- Tagline: "The AI brain behind last-mile logistics"

## Slide 2 — The Problem (0:20–1:00)
- 4 pain points with numbers:
  - Demand unpredictability → 15% fleet over-cost
  - Inefficient routing → SGD 2M/year in excess fuel & overtime
  - Fraud claims → SGD 1.5M/year in losses
  - Repetitive CS tickets → 10,000/day at SGD 6 each
- Why NOW: SEA e-commerce growth driving volume complexity

## Slide 3 — Our Approach (1:00–1:30)
- 5 problems selected: P1 Demand, P2 Routing, P5 Fraud, P6 Chatbot, P10 Control Tower
- All 6 AI types covered: ML, Deep Learning, GenAI, RAG, Agentic AI, Optimization
- Differentiator: we built the backend intelligence, not the driver app

## Slide 4 — Architecture Diagram (1:30–2:30)
- Full system diagram: data → ML models → LangGraph agents → FastAPI dashboard
- Hosting: HF Spaces (Docker) + Gemini API + Groq API
- Highlight: LangGraph as the orchestration brain for both Control Tower AND Chatbot

## Slide 5 — Problem 1: Demand Forecasting (2:30–3:00)
- Pain: unpredictable volume spikes destroy fleet planning
- Solution: LSTM + Prophet ensemble with marketing spend & sale calendar features
- Result: MAPE reduced from 25% to ~12%

## Slide 6 — Problem 2: Route Optimization (3:00–3:30)
- Pain: static routing misses spikes; bad weather compounds delays
- Solution: TSP Nearest-Neighbour over 28 Singapore postal districts + live weather factor (Open-Meteo)
- Result: Sub-second dispatch decisions; 39% reduction in failed first-attempt deliveries

## Slide 7 — Problem 5: Fraud Detection (3:30–4:00)
- Pain: false claims eroding margins; manual review doesn't scale
- Solution: Isolation Forest (unsupervised) + LightGBM (supervised) + STP + dynamic thresholding
- Result: 75% fraud detection rate vs 20% manual; 84% reduction in human review workload

## Slide 8 — Problem 6: Agentic RAG Chatbot (4:00–5:00)
- Pain: 10,000 repetitive CS tickets/day; multi-part questions get generic answers
- Solution: LangGraph 3-stage pipeline
  - **Decomposer** (Gemini 2.5 Flash): splits multi-question messages into sub-questions
  - **Processor**: 6 specialist agents — 5 with dedicated ChromaDB collections; Escalation uses direct Gemini; DuckDuckGo + Gemini 2.5 Flash web fallback when no docs found (Groq activates only on rate-limit)
  - **Synthesizer**: merges answers with per-agent section headers
- Result: 42% ticket deflection; multi-question queries fully resolved in one response; zero dead-ends (web fallback ensures an answer is always given)

## Slide 9 — Problem 10: Multi-Agent Control Tower (5:00–5:30)
- Pain: siloed ops teams react sequentially, not simultaneously
- Solution: LangGraph 5-agent sequential pipeline over shared `TowerState` — Demand → Route → Warehouse → Pricing → Customer → Coordinator; Fraud Scanner and RAG Chatbot also adapt automatically via shared `global_demand_volume`
- Result: 4-hour decision lag → 2 minutes; live demo shows 5 LangGraph agents coordinating in under 0.05 seconds

## Slide 10 — Live Demo (5:30–7:30)
- Open the dashboard
- Trigger an 11.11 spike simulation → show Control Tower coordination logs
- Switch to Route Optimizer → show SG district map with dynamic rider dispatch
- Switch to Fraud Scanner → scan parcels, then select a specific Claim ID to show single-dot isolation on the signal map; adjust threshold slider to filter flagged claims
- Switch to Chatbot → ask a multi-question: "Where is my parcel NV-100125? Also, how do I file a claim?"
- Show: two agent badges (📦 Tracking + 📋 Claims), debug trace with source attribution

## Slide 11 — Deployment Strategy (7:30–8:15)
- Phase 1 (now): Docker on HF Spaces, Gemini + Groq APIs, ChromaDB auto-built at startup
- Phase 2: Modal/Cloud Run, LangSmith monitoring, real NinjaVan data
- Phase 3: GCP Kubernetes, Pinecone managed vector DB, enterprise LLM tier

## Slide 12 — ROI Summary (8:15–8:45)
- Table: SGD 17.8M total annual savings
- Payback period: < 1 month (chatbot deflection alone covers costs)
- Year 1 ROI: ~1,300%

## Slide 13 — Risk & Ethics (8:45–9:15)
- Top risks: routing bias, driver surveillance, chatbot web-search accuracy
- Mitigations: Differential Privacy, Human-in-the-Loop, ChromaDB-first policy, escalation path
- PDPA compliance

## Slide 14 — Conclusion (9:15–10:00)
- What we built: 5 AI modules covering all 6 required AI types
- Key differentiators: multi-agent chatbot with query decomposition + web fallback; real-time global state across all agents
- Next steps: pilot with real NinjaVan data at one regional hub
- Q&A
