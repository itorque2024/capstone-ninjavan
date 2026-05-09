"""Retrain fraud model on enriched dataset with real behavioural features."""
import os, warnings, sys
warnings.filterwarnings("ignore")
_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
os.chdir(_ROOT)
sys.path.insert(0, _ROOT)

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, f1_score
from lightgbm import LGBMClassifier
from src.models.wrappers import FraudDetectionModel

raw = pd.read_csv("ninjavan_optionB_datasets/fraud_data.csv")
print(f"Shape: {raw.shape} | Fraud rate: {raw['fraud_flag'].mean()*100:.2f}%")

df = raw.copy()
df["value_log"]     = np.log1p(df["parcel_value"])
df["value_zscore"]  = (df["parcel_value"] - df["parcel_value"].mean()) / df["parcel_value"].std()
df["value_pct"]     = df["parcel_value"].rank(pct=True)
df["is_high_value"] = (df["parcel_value"] > df["parcel_value"].quantile(0.95)).astype(int)
df["is_very_low"]   = (df["parcel_value"] < df["parcel_value"].quantile(0.02)).astype(int)

FEATURES = [
    "parcel_value", "value_log", "value_zscore", "value_pct",
    "is_high_value", "is_very_low",
    "prior_claims", "account_age_days", "claim_lag_days",
]

TRAIN_STATS = {
    "value_mean": float(df["parcel_value"].mean()),
    "value_std":  float(df["parcel_value"].std()),
    "p95":        float(df["parcel_value"].quantile(0.95)),
    "p02":        float(df["parcel_value"].quantile(0.02)),
}

X = df[FEATURES]
y = df["fraud_flag"]

iso_forest = IsolationForest(n_estimators=200, contamination=float(y.mean()), random_state=42, n_jobs=-1)
iso_forest.fit(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scale_pos = (y_train == 0).sum() / (y_train == 1).sum()

lgbm = LGBMClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.05, num_leaves=31,
    scale_pos_weight=scale_pos, random_state=42, n_jobs=-1, verbose=-1,
)
lgbm.fit(X_train, y_train, eval_set=[(X_test, y_test)], callbacks=[])
lgbm_prob = lgbm.predict_proba(X_test)[:, 1]
print(f"ROC-AUC: {roc_auc_score(y_test, lgbm_prob):.4f}")

best_t, best_f1 = 0.5, 0.0
for t in np.arange(0.2, 0.8, 0.01):
    preds = (lgbm_prob >= t).astype(int)
    f1 = f1_score(y_test, preds, pos_label=1, zero_division=0)
    if f1 > best_f1:
        best_f1, best_t = f1, t

print(f"Optimal threshold: {best_t:.2f} → Fraud F1: {best_f1:.3f}")
print(classification_report(y_test, (lgbm_prob >= best_t).astype(int), target_names=["Legit", "Fraud"]))


fraud_model = FraudDetectionModel(iso_forest, lgbm, FEATURES, TRAIN_STATS, best_t)

# Sanity check
test = pd.DataFrame([
    {"parcel_value": 25.0,  "prior_claims": 0, "account_age_days": 500, "claim_lag_days": 10},
    {"parcel_value": 450.0, "prior_claims": 6, "account_age_days": 5,   "claim_lag_days": 1},
])
print(f"Normal: score={fraud_model.decision_function(test)[0]:.3f} flag={fraud_model.predict_fraud(test)[0]}")
print(f"Fraud:  score={fraud_model.decision_function(test)[1]:.3f} flag={fraud_model.predict_fraud(test)[1]}")

os.makedirs("src/models", exist_ok=True)
joblib.dump(fraud_model, "src/models/fraud_model.pkl")
print("Saved → src/models/fraud_model.pkl")
