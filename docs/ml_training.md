# Machine Learning — Training Guide

Covers the three ML models in the Operations Intelligence Suite: Demand Forecasting, Predictive Maintenance, and Fraud Detection. Includes dataset descriptions, model architecture, how to train on Google Colab, and evaluation results.

---

## Overview

| # | Problem | Notebook | Model Type | Output |
|---|---------|----------|-----------|--------|
| 1 | Demand Forecasting | `01_demand_forecasting.ipynb` | LSTM + Prophet ensemble | `demand_model.pkl`, `demand_scaler.pkl` |
| 2 | Predictive Maintenance | `02_predictive_maintenance.ipynb` | XGBoost classifier | `maintenance_model.pkl` |
| 3 | Fraud Detection | `03_fraud_detection.ipynb` | Isolation Forest + LightGBM | `fraud_model.pkl` |

All models are saved as `.pkl` wrapper files to `src/models/` and committed to GitHub so Hugging Face Spaces can use them without retraining.

---

## Datasets

All datasets are synthetic but statistically realistic, designed to mirror real NinjaVan logistics patterns. They live in `ninjavan_optionB_datasets/` and are committed to the repo — no external downloads needed.

### `demand_data.csv`
Simulates daily parcel volume across Singapore warehouses.

| Column | Description |
|--------|-------------|
| `date` | Daily timestamp |
| `warehouse_id` | Warehouse identifier |
| `shipment_volume` | Actual parcel count (target) |
| `marketing_spend` | Daily ad spend (SGD) |
| `is_sale_day` | 1 if 11.11 / 12.12 / payday sale |

### `maintenance_data.csv`
Vehicle sensor readings with engineered failure labels.

| Column | Description |
|--------|-------------|
| `vehicle_id` | Fleet vehicle identifier |
| `vehicle_health_score` | Composite health score (0–100) |
| `engine_temp_c` | Engine temperature (°C) |
| `tyre_pressure_kpa` | Tyre pressure (kPa) |
| `vibration_g` | Drivetrain vibration (g-force) |
| `km_since_service` | Distance since last service (km) |
| `at_risk` | Label: 1 = failure risk within 48–72 h |

### `fraud_data.csv`
Shipping claim records with behavioural signals.

| Column | Description |
|--------|-------------|
| `parcel_id` | Claim identifier |
| `parcel_value` | Declared value (SGD) |
| `prior_claims` | Number of prior claims from this account |
| `account_age_days` | Account age at time of claim |
| `claim_lag_days` | Days between delivery and claim |
| `fraud_flag` | Label: 1 = confirmed fraudulent claim |

---

## How to Train (Google Colab)

All three notebooks are designed to run on Google Colab with no local setup required.

### One-time steps
1. Ensure the GitHub repo is public **or** you are logged in to GitHub on Colab
2. Set your Colab runtime: **Runtime → Change runtime type**
   - Notebook 01: select **GPU** (speeds up LSTM training)
   - Notebooks 02 & 03: **CPU** is sufficient

### Running a notebook

1. Open the notebook directly via the **Open in Colab** badge at the top of each file on GitHub
2. Run all cells top to bottom (**Runtime → Run all**)
3. The last cell saves the `.pkl` to `src/models/` inside the cloned repo on Colab
4. Download the updated `.pkl` from the Colab file panel (`/content/capstone-ninjavan/src/models/`)
5. Replace the local file and commit:
   ```bash
   git add src/models/<model>.pkl
   git commit -m "chore: retrain <model>"
   git push origin main
   ```

### What each notebook does

#### `01_demand_forecasting.ipynb`
```
Load demand_data.csv
→ Engineer features (lag, rolling avg, sale flags)
→ Scale with MinMaxScaler → save demand_scaler.pkl
→ Train LSTM (3-layer, 128 units) on 80% of data
→ Train Prophet with marketing_spend as regressor
→ Ensemble: average LSTM + Prophet predictions
→ Wrap in DemandForecastModel → save demand_model.pkl
```
- GPU recommended; LSTM training takes ~5 min on Colab GPU, ~25 min on CPU

#### `02_predictive_maintenance.ipynb`
```
Load maintenance_data.csv
→ Engineer health_band categorical feature
→ Apply SMOTE to balance the at_risk minority class
→ Train XGBoost classifier (200 estimators, max_depth=5)
→ Tune threshold for recall > 0.85 (catching near-misses matters more than precision)
→ Wrap in MaintenanceRiskModel → save maintenance_model.pkl
```
- Training takes ~2 min on CPU

#### `03_fraud_detection.ipynb`
```
Load fraud_data.csv
→ Engineer value-based features (log, z-score, percentile, high/low flags)
→ Train Isolation Forest (unsupervised anomaly score)
→ Train LightGBM (supervised, uses fraud_flag labels)
→ Tune threshold to maximise F1 on the fraud class
→ Wrap both in FraudDetectionModel → save fraud_model.pkl
```
- Training takes ~3 min on CPU

---

## Model Architecture

### Demand Forecasting — LSTM + Prophet Ensemble

```
Input: [shipment_volume_lag7, marketing_spend, is_sale_day, ...]
         │
   MinMaxScaler
         │
   LSTM Layer 1 (128 units, return_sequences=True)
         │
   Dropout (0.2)
         │
   LSTM Layer 2 (64 units, return_sequences=True)
         │
   Dropout (0.2)
         │
   LSTM Layer 3 (32 units)
         │
   Dense (1) → LSTM forecast
         │
   ┌─────┴──────┐
   │            │
LSTM pred  Prophet pred  ← (seasonality + marketing regressor)
   │            │
   └────avg─────┘
         │
   Ensemble output (daily volume)
```

**Wrapper:** `DemandForecastModel.predict(horizon)` returns `{"baseline_avg": float, "values": [float, ...]}`

### Predictive Maintenance — XGBoost

```
Input: [vehicle_health_score, engine_temp_c, tyre_pressure_kpa,
        vibration_g, km_since_service, health_band]
         │
   SMOTE (oversample minority at_risk=1 class)
         │
   XGBoostClassifier (n_estimators=200, max_depth=5, lr=0.05)
         │
   predict_proba → failure probability (0–1)
```

**Wrapper:** `MaintenanceRiskModel.predict_proba(df)` — accepts a DataFrame with only `vehicle_health_score`; imputes remaining sensor features from that score automatically.

### Fraud Detection — Isolation Forest + LightGBM

```
Input: [parcel_value, prior_claims, account_age_days, claim_lag_days,
        + derived: value_log, value_zscore, value_pct, is_high_value, is_very_low]
         │
   ┌─────┴──────────────────────────┐
   │                                │
Isolation Forest              LightGBM Classifier
(anomaly score → −1/+1)      (fraud probability 0–1)
   │                                │
   └─────────────┬──────────────────┘
                 │
          Combined score
                 │
         Dynamic threshold
                 │
        fraud / legitimate
```

**Wrapper:** `FraudDetectionModel.decision_function(df)` — accepts a DataFrame with only `parcel_value`; imputes remaining features automatically.

---

## Evaluation Results

### Demand Forecasting

| Metric | Baseline (no ML) | Our Model |
|--------|-----------------|-----------|
| MAPE | ~25% | ~12% |
| Sale-day spike error | >40% | <15% |

### Predictive Maintenance

| Metric | Score |
|--------|-------|
| ROC-AUC | ~0.93 |
| Recall (at-risk vehicles caught) | >85% |
| Precision | ~78% |
| 5-fold CV Recall | stable ±3% |

High recall is prioritised over precision — missing a failing vehicle (false negative) costs far more than a false alarm.

### Fraud Detection

| Metric | Isolation Forest | LightGBM | Combined |
|--------|-----------------|----------|---------|
| Fraud detection rate | ~55% | ~75% | ~75% |
| False positive rate | ~15% | ~8% | ~8% |
| Manual review reduction | — | — | ~84% |

Compared to manual review baseline of ~20% detection rate.

---

## Retraining

Retrain when:
- New data significantly changes the distribution (e.g., real NinjaVan data replaces synthetic)
- Model performance degrades in production (MAPE drifts above 20%, recall drops below 80%)
- New features are added to the dataset

Retrain steps:
1. Update the dataset in `ninjavan_optionB_datasets/`
2. Open the relevant notebook in Colab and run all cells
3. Download the new `.pkl` file and replace in `src/models/`
4. Commit and push — HF Spaces auto-deploys on next push

For Phase 2 onwards, scheduled monthly retraining via Modal/Cloud Run is recommended (see `deployment_strategy.md`).

---

## Model Wrappers (`src/models/wrappers.py`)

The pkl files are not raw sklearn/XGBoost models — they are custom wrapper classes that handle missing features gracefully:

| Wrapper | Key method | Handles |
|---------|-----------|---------|
| `DemandForecastModel` | `predict(horizon)` | Returns dict with `baseline_avg` + `values` list |
| `MaintenanceRiskModel` | `predict_proba(df)` | Imputes sensor columns from `vehicle_health_score` alone |
| `FraudDetectionModel` | `decision_function(df)` | Imputes all features from `parcel_value` alone |

This design means the dashboard can call agents with minimal input — a single health score or parcel value is enough for real-time inference.

---

## Known Limitations & Future Improvements

### Fraud Detection

**Current scope:** The model detects fraudulent *claim behaviour* — patterns like high parcel value + new account + fast claim submission. It works well for false missing-parcel claims filed through the customer support channel.

**What it does not cover:** Driver-side theft, where a driver marks a parcel as "Delivered" in the system but keeps it. This type of fraud leaves no claim signal — it requires a different data layer entirely:

| Improvement | Data Required | Approach |
|-------------|--------------|---------|
| Driver delivery fraud | GPS trace + delivery photo metadata | Anomaly detection on route deviation and photo timestamp gaps |
| Address manipulation | Delivery address change logs | Flag last-minute address changes on high-value parcels |
| Collusion rings | Network graph of drivers + addresses | Graph-based fraud detection (GNN or community detection) |

**For Phase 2** with real NinjaVan data, driver GPS traces and photo-on-delivery logs would unlock these additional fraud vectors.

### Demand Forecasting

**Current scope:** Forecasts daily parcel volume at the regional level (ID/TH/MY). Marketing spend and sale flags are synthetically generated in the notebook.

**What it does not cover:** Warehouse-level granularity with real marketing calendar data, or external shocks (port strikes, COVID-style disruptions). Real NinjaVan marketing spend data would significantly improve spike prediction accuracy.

### Predictive Maintenance

**Current scope:** Predicts vehicle failure risk from sensor readings. Sensor data is synthetic but correlated with `vehicle_health_score`.

**What it does not cover:** Real OBD-II telematics feeds, tyre wear from mileage curves, or fleet-level scheduling (knowing vehicle A needs service doesn't automatically re-route its parcels). Integration with the route optimizer in Phase 2 would close this loop.
