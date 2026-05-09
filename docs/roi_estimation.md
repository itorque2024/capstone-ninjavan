# ROI Estimation

## Assumptions
- Fleet size: 1,000 trucks (Singapore operations)
- Daily parcel volume: 500,000 parcels (Singapore operations; NinjaVan group-wide is ~1–2M/day across 6 SEA markets)
- Annual CS tickets: 3.6M (10,000/day)
- Average fraud claim value: SGD 150
- Annual claims volume: 200,000 claims
- Fraud rate: 5% of annual claims volume (10,000 fraudulent claims/year)
- CS ticket cost: SGD 6/ticket — blended outsourced rate (fully-loaded in-house Singapore agent = SGD 11–20/ticket)
- Failed first-attempt rate: 18% — conservative Singapore-specific estimate (SEA regional average is 20–40% due to COD dominance, addressing gaps)
- Model performance benchmarks from Gartner, McKinsey logistics AI reports (2024–2025)

---

## Problem 1 — Demand Forecasting

| Metric | Baseline | With AI | Improvement |
|--------|----------|---------|-------------|
| Forecast error (MAPE) | ~25% | ~12% | -52% |
| Fleet over-provisioning cost | SGD 5M/year | SGD 2M/year | -SGD 3M/year |
| SLA breach penalty | SGD 1.5M/year | SGD 0.8M/year | -SGD 700K/year |
| **Total annual saving** | | | **~SGD 3.7M** |

---

## Problem 2 — Intelligent Route Optimization

| Metric | Baseline | With AI | Improvement |
|--------|----------|---------|-------------|
| Failed first-attempt deliveries | ~18% of parcels | ~11% | -39% |
| Routing inefficiency overhead (excess mileage + redelivery overtime) | SGD 8M/year | SGD 6M/year | -SGD 2M/year |
| Routing decision latency | ~2 hours (manual) | <1 second (automated) | -99.99% |
| **Total annual saving** | | | **~SGD 2M** |

> Note: Total fleet fuel + maintenance budget for 1,000 SG trucks is ~SGD 24–45M/year (SGD 24–54K/truck). The SGD 8M baseline above represents only the routing-related waste component (~25–33% of total fuel budget attributable to suboptimal paths and failed-delivery returns), not the full fleet operations cost. The SGD 2M saving is therefore a conservative estimate (~8% of total fuel spend).

---

## Problem 5 — Fraud Detection

| Metric | Baseline | With AI | Improvement |
|--------|----------|---------|-------------|
| Fraud detection rate | ~20% (manual review) | ~75% | +55% |
| Annual fraudulent claim losses | SGD 1.5M | SGD 375K | -SGD 1.125M |
| Claims reviewed per human agent/day | 50 | 8 (only high-risk) | -84% human workload |
| **Total annual saving** | | | **~SGD 1.1M** |

---

## Problem 6 — Agentic RAG Customer Chatbot

| Metric | Baseline | With AI | Improvement |
|--------|----------|---------|-------------|
| Tickets handled by human agents | 10,000/day | 5,800/day | -42% deflection |
| Cost per human ticket | SGD 6 | ~SGD 0.11 (AI) | -98% |
| Multi-question resolution rate | ~30% (one topic per ticket) | ~90% (decomposer handles all sub-questions) | +200% |
| Average handle time (complex queries) | 8 min | <10 seconds | -98% |
| **Daily saving** | SGD 60,000 | SGD 35,280 | **SGD 24,720/day** |
| **Annual saving** | | | **~SGD 9M** |

---

## Problem 10 — Multi-Agent Control Tower

| Metric | Baseline | With AI | Improvement |
|--------|----------|---------|-------------|
| Ops decision latency | ~4 hours (manual cross-team) | ~2 minutes (automated) | -98% |
| Ops efficiency gain | — | 15% | +15% |
| **Estimated annual value** | | | **~SGD 2M** (ops cost reduction) |

---

## Summary

| Problem | Annual Saving (SGD) |
|---------|-------------------|
| P1 — Demand Forecasting | 3,700,000 |
| P2 — Route Optimization | 2,000,000 |
| P5 — Fraud Detection | 1,100,000 |
| P6 — Agentic RAG Chatbot | 9,000,000 |
| P10 — Control Tower | 2,000,000 |
| **Total** | **~SGD 17.8M/year** |

**Estimated implementation cost (Year 1):** SGD 800K–1.2M (team, cloud infra, API costs)
**Payback period:** < 1 month (driven by chatbot deflection savings)
**Year 1 ROI:** ~1,300%
