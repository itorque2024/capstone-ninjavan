# NinjaVan Operations Intelligence Suite — AI Video Generator Script

**Instructions for Video Generator (HeyGen, Synthesia, etc.):**
*   **Avatar:** Professional, modern business attire.
*   **Tone:** Confident, analytical, industry-expert.
*   **Pacing:** Deliberate, with slight pauses between major sections.
*   **Target Length:** ~8 to 10 minutes (approx. 1,200 words).

---

### [0:00 - 1:30] Introduction & Overarching Problem
**[Visual Cue: Avatar centered. Background shows a chaotic warehouse or traffic congestion. Text overlay: "The Silo Problem in Logistics"]**

**[Spoken Script:]**
Hello, and welcome to our Capstone Presentation. Today, we are presenting the NinjaVan Operations Intelligence Suite — a fully integrated, Multi-Agent AI architecture.

The modern logistics industry faces a critical problem: Siloed Operations. When a massive event happens — like an 11.11 mega-sale or extreme monsoon weather — the operational shock waves are felt everywhere. However, departments like Demand Planning, Fleet Routing, Fraud Detection, and Customer Service operate using disconnected data.

This fragmentation causes severe reaction delays, vehicle shortages, and overwhelmed customer support. Ultimately, it destroys the single most important metric in logistics: the First-Attempt Success Rate. When a delivery fails and returns to the hub, the profit margin for that parcel is entirely wiped out.

To solve this, we did not just build five isolated machine learning models. We built an Agentic Control Tower that acts as a single, shared brain — allowing five distinct AI specialists to communicate in milliseconds.

---

### [1:30 - 3:00] Problem 1: Demand Forecasting
**[Visual Cue: Show the Demand Forecasting tab in the dashboard. Highlight the 11.11 spike graph.]**

**[Spoken Script:]**
The chronological life of a package begins with our first module: Demand Forecasting.

Manual planning leads to either over-provisioning trucks — wasting fuel — or under-provisioning, which guarantees SLA breaches. We solved this using a Machine Learning Ensemble combining Long Short-Term Memory Neural Networks and Prophet.

This architecture captures long-range, non-linear dependencies while natively handling weekly and annual seasonality. More importantly, we injected exogenous features — marketing spend and holiday calendars — into the model. As you can see on the screen, when we simulate the 11.11 sale, our AI predicts the massive volume spike before it happens, allowing NinjaVan to prepare the exact fleet capacity required. Forecast error drops from 25% to approximately 12%.

---

### [3:00 - 4:30] Problem 2: Intelligent Route Optimization
**[Visual Cue: Show the Route Optimizer tab with the Singapore Mapbox map and district route paths.]**

**[Spoken Script:]**
Once packages are ready for dispatch, they must reach customers efficiently. Our second module is Intelligent Route Optimization.

Traditional routing software is static and cannot react to the dynamic volume spikes our Demand model predicts. Deep Reinforcement Learning is popular in academia, but is far too slow for real-time dispatch at scale.

Instead, we designed a TSP Nearest-Neighbor heuristic operating across all 28 Singapore postal districts. The system calculates the optimal rider dispatch sequence in under one second. It also pulls live rain data from Open-Meteo to dynamically adjust travel time estimates — because a 40mm rainfall day changes everything for a delivery rider.

When our Demand AI predicts a surge of thirty thousand packages, the Routing AI instantly reads that state and calculates the exact number of riders needed, plotting the fastest geographic paths across Singapore. This directly protects the First-Attempt Success Rate.

---

### [4:30 - 6:00] Problem 5: Fraud Detection & Claims
**[Visual Cue: Show the Fraud Detection tab, highlighting the "Auto-Approved by AI (STP)" metrics and risk scatter plot.]**

**[Spoken Script:]**
Despite our best routing efforts, the reality of the industry is that operational pressure sometimes leads to packages being left unattended, resulting in damaged or stolen goods and an influx of customer claims.

Our third module tackles Fraud Detection. Instead of humans reading thousands of claims, we implemented Straight-Through Processing powered by an Unsupervised Isolation Forest model combined with a supervised LightGBM classifier.

This AI instantly isolates mathematically abnormal claims. As shown on the dashboard, it automatically approves the ninety percent of claims that are statistically normal — saving massive human labor — and routes only the high-risk anomalies to a human agent for review.

Furthermore, the AI uses dynamic thresholding. Because it shares data with the Demand module, it knows when operations are chaotic during a mega-sale and automatically relaxes its strictness — because legitimate damage is statistically more likely during those periods.

---

### [6:00 - 7:30] Problem 6: Agentic RAG Customer Service Chatbot
**[Visual Cue: Show the CX Chatbot tab. Type a multi-question: "Where is my parcel NV-100001? Also how do I file a damage claim?" Show two agent badges appearing in the response.]**

**[Spoken Script:]**
When packages are delayed, customers reach out. Our fourth module is an AI-Powered Customer Service Chatbot — but not a simple one.

Most chatbots fail on complex multi-part questions. A customer asking "Where is my parcel, and how do I file a claim?" gets one generic answer that addresses neither properly. Our system is fundamentally different.

We built a three-stage LangGraph pipeline. First, a Decomposer — powered by Gemini 2.5 Flash — reads the customer message and intelligently splits it into individual sub-questions, tagging each with the correct intent: tracking, delivery, claims, policy, operations, or escalation.

Second, a Processor routes each sub-question to the right specialist agent. Each of our six agents — Tracking, Delivery, Claims, Policy, Ops, and Escalation — queries a dedicated ChromaDB collection. Five specialist collections cover the domain knowledge; the Ops Agent shares the Delivery collection for operational queries, and the Escalation Agent goes directly to Gemini for empathetic responses without a knowledge base lookup. If the knowledge base has no relevant documents for a question, the system automatically triggers a DuckDuckGo web search and passes those results to Gemini 2.5 Flash for a grounded answer. If Gemini hits a rate limit, Groq's Llama 3.3 steps in as an automatic fallback. The customer never hits a dead end.

Third, a Synthesizer combines all the sub-answers into one coherent response, with clear per-agent section headers. You can see on screen that our multi-question example produced two expert answers — one from the Tracking Agent with the real parcel status, and one from the Claims Agent with step-by-step claim instructions.

Because the chatbot also reads the global demand state, it knows during an 11.11 surge to proactively mention that delivery SLAs are extended — before the customer even asks about delays.

---

### [7:30 - 9:00] Problem 10: Multi-Agent Control Tower
**[Visual Cue: Show the architecture diagram. Then show the Control Tower tab with the simulation logs.]**

**[Spoken Script:]**
How do all these modules work together? This brings us to our fifth and final module: The Multi-Agent Control Tower, powered by LangGraph.

LangGraph allows us to build stateful agent workflows where all five agents share a single global state called `TowerState`. The pipeline runs in sequence: the Demand Agent forecasts the 11.11 spike and writes the peak volume to shared state. The Route Agent reads this and instantly calculates how many riders are needed. The Warehouse Agent reassigns picking zones for higher throughput. The Pricing Agent activates surge pricing based on the demand ratio. Finally, the Customer Agent drafts proactive delay alerts for affected customers.

The Coordinator then reviews all five decisions, flags any conflicts — for example, if surge pricing is active at the same time as long delivery windows — and calculates the total decision latency.

Additionally, the Fraud Scanner and RAG Chatbot — which run as standalone modules — automatically read the same demand volume via the API. The fraud threshold relaxes during the spike because legitimate damage claims statistically rise during mega-sales. The chatbot activates delay-warning mode before customers even ask about their orders.

This entire workflow — five coordinated agents producing a complete operational plan — completes in under zero point zero five seconds, compared to the four-hour lag of manual cross-team communication.

---

### [9:00 - 10:00] Risks, Ethics & Conclusion
**[Visual Cue: Bullet points on screen: "Algorithmic Fairness", "Differential Privacy", "Human-in-the-Loop", "Chatbot Transparency".]**

**[Spoken Script:]**
As we deploy these technologies, we must address the ethical risks.

First is algorithmic bias. We rigorously audit our Routing model to ensure it does not bypass high-congestion, lower-income neighborhoods in the pursuit of efficiency, preventing service inequality.

Second is driver surveillance. We implement Differential Privacy on fleet GPS data and enforce a strict Human-in-the-Loop policy. The Control Tower acts in an advisory capacity only — no AI metric is used for automatic disciplinary action.

Third is chatbot transparency. Every chatbot response shows which specialist agent answered and whether the answer came from our verified knowledge base, a web search, or general AI knowledge. Customers always have an escalation path to a human agent.

By breaking down data silos, our Operations Intelligence Suite proves that when predictive, prescriptive, and generative AI communicate through a centralized Control Tower, a logistics company can protect its profit margins, dramatically elevate the customer experience, and save an estimated SGD 17.8 million annually.

Thank you for watching.
