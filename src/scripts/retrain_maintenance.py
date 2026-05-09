"""Retrain maintenance model on enriched dataset with real sensor columns."""
import os, warnings, sys
warnings.filterwarnings("ignore")
_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
os.chdir(_ROOT)
sys.path.insert(0, _ROOT)

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from src.models.wrappers import MaintenanceRiskModel

raw = pd.read_csv("ninjavan_optionB_datasets/maintenance_data.csv")
print(f"Shape: {raw.shape}")
print(f"At-risk rate: {raw['at_risk'].mean()*100:.1f}%")
print(f"Columns: {list(raw.columns)}")

df = raw.copy()
df["engine_temp_c"]     = df["engine_temp_c"].clip(80, 210)
df["tyre_pressure_kpa"] = df["tyre_pressure_kpa"].clip(145, 265)
df["vibration_g"]       = df["vibration_g"].clip(0, 2.5)
df["km_since_service"]  = df["km_since_service"].clip(0, 15000)
df["health_band"] = pd.cut(
    df["vehicle_health_score"],
    bins=[0, 40, 60, 80, 100],
    labels=[3, 2, 1, 0],
).astype(int)

FEATURES = [
    "vehicle_health_score", "engine_temp_c", "tyre_pressure_kpa",
    "vibration_g", "km_since_service", "health_band",
]

X = df[FEATURES]
y = df["at_risk"]
print(f"Class balance: {y.value_counts().to_dict()}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

xgb = XGBClassifier(
    n_estimators=200, max_depth=5, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    eval_metric="logloss", random_state=42,
)
xgb.fit(X_train_res, y_train_res, eval_set=[(X_test, y_test)], verbose=50)

y_pred = xgb.predict(X_test)
y_prob = xgb.predict_proba(X_test)[:, 1]
print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")
print(classification_report(y_test, y_pred, target_names=["Healthy", "At Risk"]))


maintenance_model = MaintenanceRiskModel(xgb, FEATURES)
test_input = pd.DataFrame([{"vehicle_health_score": 45.0}, {"vehicle_health_score": 85.0}])
probs = maintenance_model.predict_proba(test_input)[:, 1]
print(f"Sanity: score=45 → {probs[0]:.3f} | score=85 → {probs[1]:.3f}")

os.makedirs("src/models", exist_ok=True)
joblib.dump(maintenance_model, "src/models/maintenance_model.pkl")
print("Saved → src/models/maintenance_model.pkl")
