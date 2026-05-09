# Risk & Ethics

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| Fraud model incorrectly flags legitimate customers | Medium | High | Human-in-the-loop review before any claim rejection; mandatory appeal process |
| Algorithmic bias in routing | Medium | High | Audit route optimizer to ensure it does not systematically deprioritize high-congestion (often lower-income) neighborhoods, causing service inequality |
| Excessive driver surveillance | High | Medium | GPS tracking for "optimization" can feel like "Big Brother"; implement Differential Privacy on driver data; ban AI metrics from automatic disciplinary action |
| LSTM demand model fails on black swan events | Low | High | Uncertainty bounds on forecasts; automatic fallback to rule-based planning when model confidence is low |
| RAG chatbot hallucinates information | Low | Medium | Answers grounded in retrieved ChromaDB documents; if no relevant docs found, web search used as fallback with source attribution; uncertain queries escalate to human |
| Chatbot web search returns outdated or inaccurate content | Medium | Medium | DuckDuckGo results used only when ChromaDB has zero matching docs; Groq/Gemini instructed to caveat answers based on web results; escalation path always available |
| Customer PII exposed in training data | Low | Very High | Anonymize all training data; encrypt ChromaDB at rest; comply with PDPA |
| Over-reliance on AI decisions | Medium | High | All high-stakes decisions require human approval; AI recommends, human decides |
| Gemini API rate limits during peak events | Medium | Medium | joblib.Memory cache for repeated identical queries; retry logic with exponential backoff on 429/503; Groq as secondary LLM |

---

## Ethical Considerations

### Fairness & Algorithmic Bias
- **Service Inequality:** The Intelligent Routing model must be audited to ensure it does not bypass or deprioritize high-congestion, lower-income neighborhoods in its pursuit of mathematical efficiency.
- **Fraud Bias:** Regular fairness audits must be conducted using disaggregated metrics to ensure the Isolation Forest does not unfairly flag specific geographic regions or customer demographics.

### Transparency
- All AI-assisted decisions must be explainable. LightGBM SHAP values provide feature-level explanation for fraud flags.
- Customers are informed when interacting with an AI chatbot. The chatbot badge (📦 Tracking Agent, 📋 Claims Agent, etc.) shows which specialist is answering.
- The "Agent Trace" debug log is available in the UI, showing which agents handled the query, which data source was used (chromadb+gemini / web+gemini / web+groq [rate-limit only]), and how many documents were retrieved.

### Data Privacy & Surveillance
- **Driver Surveillance (Big Brother):** Constant GPS tracking of fleet drivers for route optimization can infringe on worker privacy. Differential Privacy must be implemented on driver data, and AI efficiency metrics must be explicitly prohibited from automatic disciplinary action.
- **Customer Privacy:** Customer data used for training is anonymized. The RAG knowledge base contains only public-facing NinjaVan policies — no customer PII.
- Comply with Singapore PDPA, Thailand PDPA, and applicable SEA data protection regulations.

### Human Oversight
- The Control Tower operates in **advisory mode** — it surfaces alerts and recommendations; operational staff make final decisions.
- The chatbot Escalation Agent always provides a human follow-up path (1800-NJV-CARE / escalation@ninjavan.co) for sensitive situations.
- An audit log of all agent decisions is maintained for accountability and post-incident review.

---

## Responsible AI Checklist

- [ ] Training data reviewed for bias and representativeness
- [ ] Model explainability implemented (SHAP for LightGBM fraud model)
- [ ] Human-in-the-loop for all high-stakes decisions (fraud claims, escalations)
- [ ] PII anonymization verified before training
- [ ] Chatbot escalation path tested and working (Escalation Agent → human handoff)
- [ ] Web search fallback caveated appropriately (source: web+gemini noted in debug log; web+groq shown only when Gemini rate-limited)
- [ ] Gemini API rate limit handling in place (retry + joblib cache)
- [ ] Model monitoring and drift detection planned for Phase 2
- [ ] Ethics review completed before production deployment
