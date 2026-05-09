"""
Fraud Detection Agent — decision-driving outputs for fraud ops teams.

Per claim returns:
  risk_score (0-1), risk_pct, fraud_pattern, why_flagged (human explanation),
  action (BLOCK/REVIEW/MONITOR/CLEAR), priority_score, iso_anomaly_score

Summary returns:
  total, flagged, flag_rate_pct, total_at_risk_sgd, high_confidence_count,
  pattern_breakdown, avg_flagged_value, avg_legit_value
"""
import joblib
import pandas as pd
from pathlib import Path

MODEL_PATH = Path(__file__).parent.parent / "models" / "fraud_model.pkl"

# ── Thresholds ─────────────────────────────────────────────────────────────────
_BLOCK   = 0.80
_REVIEW  = 0.60
_MONITOR = 0.40


def _classify_pattern(c: dict) -> tuple[str, str]:
    """
    Return (pattern_code, pattern_label) for the most prominent fraud signal.
    Rules match the 3 patterns used to generate the training data.
    """
    val  = float(c.get("parcel_value", 0))
    age  = float(c.get("account_age_days", 999))
    prv  = float(c.get("prior_claims", 0))
    lag  = float(c.get("claim_lag_days", 99))

    patterns = []
    if val > 150 and age < 30:
        patterns.append("A")
    if prv >= 4:
        patterns.append("B")
    if lag <= 2 and val > 100:
        patterns.append("C")

    if len(patterns) > 1:
        return ("MULTI", "🔥 Multiple patterns")
    if "A" in patterns:
        return ("A", "💳 High-value + New account")
    if "B" in patterns:
        return ("B", "🔁 Repeat claimer")
    if "C" in patterns:
        return ("C", "⚡ Immediate high-value claim")
    return ("ANOMALY", "🔍 Statistical anomaly")


def _explain(c: dict, risk: float) -> list[str]:
    """Build a list of human-readable reasons this claim was flagged."""
    val = float(c.get("parcel_value", 0))
    age = float(c.get("account_age_days", 999))
    prv = float(c.get("prior_claims", 0))
    lag = float(c.get("claim_lag_days", 99))

    reasons = []
    if val > 300:
        reasons.append(f"Very high claim value (S${val:.0f} — top 5% of all claims)")
    elif val > 150:
        reasons.append(f"High claim value (S${val:.0f})")

    if age < 10:
        reasons.append(f"Brand-new account ({int(age)} days old — high-risk window)")
    elif age < 30:
        reasons.append(f"Very new account ({int(age)} days old)")

    if prv >= 6:
        reasons.append(f"Habitual claimer ({int(prv)} prior claims on record)")
    elif prv >= 4:
        reasons.append(f"Repeat claimer ({int(prv)} prior claims)")

    if lag == 0:
        reasons.append("Claim filed same day as incident (immediate filing)")
    elif lag <= 2:
        reasons.append(f"Claim filed {int(lag)} day(s) after incident (unusually fast)")

    if not reasons:
        reasons.append(f"Statistical anomaly detected by Isolation Forest (score below threshold)")

    return reasons


def _action(risk: float) -> tuple[str, str]:
    if risk >= _BLOCK:
        return ("🔴 BLOCK",   "Reject claim immediately. Freeze account. Escalate to fraud team.")
    if risk >= _REVIEW:
        return ("🟠 REVIEW",  "Manual review required before any payout. Request supporting documents.")
    if risk >= _MONITOR:
        return ("🟡 MONITOR", "Approve with enhanced verification. Flag account for monitoring.")
    return      ("🟢 CLEAR",  "Auto-approve. Low fraud risk.")


def run_fraud_agent(state: dict) -> dict:
    """
    LangGraph-compatible node.
    Input state keys: claims (list[dict]), fraud_threshold (float, optional)
    Output: fraud_result with per-claim decision data + portfolio summary.
    """
    claims    = state.get("claims", [])
    threshold = state.get("fraud_threshold", _REVIEW)   # default flag at REVIEW level

    if not claims:
        return {**state, "fraud_result": {"error": "No claims data provided."}}
    if not MODEL_PATH.exists():
        return {**state, "fraud_result": {"error": "Model not trained yet. Run notebook 03."}}

    model = joblib.load(MODEL_PATH)
    df    = pd.DataFrame(claims)

    iso_scores = model.decision_function(df)
    risk_probs = model.risk_probability(df) if hasattr(model, "risk_probability") else (
        model.lgbm.predict_proba(model._prepare(df))[:, 1]
    )

    all_claims = []
    for i, c in enumerate(claims):
        risk  = float(risk_probs[i])
        iso   = float(iso_scores[i])
        val   = float(c.get("parcel_value", 0))

        pat_code, pat_label = _classify_pattern(c)
        reasons             = _explain(c, risk)
        action_badge, action_text = _action(risk)

        # Priority score: risk × claim value — what to investigate first
        priority = round(risk * val, 2)

        all_claims.append({
            "claim_id":           c.get("claim_id", i + 1),
            "parcel_value":       round(val, 2),
            "prior_claims":       int(c.get("prior_claims", 0)),
            "account_age_days":   int(c.get("account_age_days", 0)),
            "claim_lag_days":     int(c.get("claim_lag_days", 0)),
            "risk_score":         round(risk, 3),
            "risk_pct":           f"{risk * 100:.0f}%",
            "iso_anomaly_score":  round(iso, 3),
            "fraud_pattern":      pat_label,
            "pattern_code":       pat_code,
            "why_flagged":        reasons,
            "action":             action_badge,
            "action_text":        action_text,
            "priority_score":     priority,
            "is_flagged":         risk >= threshold,
        })

    flagged      = [c for c in all_claims if c["is_flagged"]]
    high_conf    = [c for c in all_claims if c["risk_score"] >= _BLOCK]
    total_at_risk = sum(c["parcel_value"] for c in flagged)
    flag_rate     = len(flagged) / max(len(all_claims), 1) * 100

    # Pattern breakdown (flagged only)
    from collections import Counter
    pattern_counts = Counter(c["pattern_code"] for c in flagged)

    avg_flagged_val = (sum(c["parcel_value"] for c in flagged) / max(len(flagged), 1))
    avg_legit_val   = (sum(c["parcel_value"] for c in all_claims if not c["is_flagged"])
                       / max(len(all_claims) - len(flagged), 1))

    # Investigation queue: sort by priority score descending
    investigation_queue = sorted(flagged, key=lambda x: x["priority_score"], reverse=True)

    return {
        **state,
        "fraud_result": {
            "all_claims":          all_claims,
            "flagged_claims":      flagged,
            "investigation_queue": investigation_queue,
            "high_confidence":     high_conf,
            "summary": {
                "total":              len(all_claims),
                "flagged":            len(flagged),
                "flag_rate_pct":      round(flag_rate, 1),
                "high_confidence":    len(high_conf),
                "total_at_risk_sgd":  round(total_at_risk, 2),
                "avg_flagged_value":  round(avg_flagged_val, 2),
                "avg_legit_value":    round(avg_legit_val, 2),
                "pattern_breakdown":  dict(pattern_counts),
            },
            "alert": (
                f"🔴 {len(high_conf)} claim(s) require IMMEDIATE blocking. "
                f"{len(flagged)} total flagged. S${total_at_risk:,.0f} at risk."
            ) if high_conf else (
                f"{len(flagged)} claim(s) flagged for review. S${total_at_risk:,.0f} at risk."
                if flagged else None
            ),
        },
    }
