"""Retrain demand forecasting model on enriched dataset with 6 SEA regions."""
import os, warnings, sys
warnings.filterwarnings("ignore")
_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
os.chdir(_ROOT)
sys.path.insert(0, _ROOT)

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error
from prophet import Prophet
from src.models.wrappers import DemandForecastModel
from src.utils.sea_holidays import get_prophet_holidays

df = pd.read_csv("ninjavan_optionB_datasets/demand_data.csv", parse_dates=["order_date"])
print(f"Shape: {df.shape}")
print(f"Regions: {sorted(df['region'].unique())}")
print(f"Date range: {df['order_date'].min().date()} → {df['order_date'].max().date()}")

# Aggregate to daily total across all 6 regions
daily = (
    df.groupby("order_date")["parcel_volume"]
    .sum()
    .reset_index()
    .rename(columns={"order_date": "ds", "parcel_volume": "y"})
    .sort_values("ds")
    .reset_index(drop=True)
)
print(f"Days in dataset: {len(daily)} | Mean daily volume: {daily['y'].mean():.0f}")

TRAIN_RATIO = 0.8
split_idx = int(len(daily) * TRAIN_RATIO)
train_df = daily.iloc[:split_idx].reset_index(drop=True)
test_df  = daily.iloc[split_idx:].reset_index(drop=True)
print(f"Train: {len(train_df)} days | Test: {len(test_df)} days")

holidays_df = get_prophet_holidays()
print(f"Holiday events loaded: {holidays_df['holiday'].nunique()} types, {len(holidays_df)} rows")

def _make_prophet():
    return Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10,
        holidays=holidays_df,
        holidays_prior_scale=15,
    )

# ── Step 1: Evaluate on train/test split (for MAPE reporting) ─────────────────
eval_model = _make_prophet()
eval_model.fit(train_df)
future_eval = eval_model.make_future_dataframe(periods=len(test_df) + 7)
prophet_fc   = eval_model.predict(future_eval)
prophet_test = (
    prophet_fc[prophet_fc["ds"].isin(test_df["ds"])][["ds", "yhat"]]
    .reset_index(drop=True)
)
prophet_pred = prophet_test["yhat"].clip(lower=0).values
actual = test_df["y"].values

def mape(a, p):
    return np.mean(np.abs((a - p) / (a + 1e-9))) * 100

print(f"Eval — MAE: {mean_absolute_error(actual, prophet_pred):.1f} | MAPE: {mape(actual, prophet_pred):.2f}%")

# ── Step 2: Retrain on FULL dataset for deployment ────────────────────────────
# Production best practice: validate on holdout, then use all data for the final model.
prophet_model = _make_prophet()
prophet_model.fit(daily)
print(f"Final model trained on full {len(daily)}-day dataset (last date: {daily['ds'].max().date()})")

# Region shares from last 90 days
df_recent = df[df["order_date"] >= df["order_date"].max() - pd.Timedelta(days=90)]
region_shares = (
    df_recent.groupby("region")["parcel_volume"].sum()
    / df_recent["parcel_volume"].sum()
).round(4).to_dict()
print(f"Region shares (last 90 days): {region_shares}")

demand_model = DemandForecastModel(
    prophet_model=prophet_model,
    scaler=None,
    history_values=daily["y"].values,
    seq_len=14,
    lstm_model=None,
    prophet_weight=1.0,
    training_end_date=str(daily["ds"].max().date()),
    region_shares=region_shares,
)
test_out = demand_model.predict(horizon=7, start_date="2026-04-29")
print(f"Sanity (7-day from 2026-04-29): baseline_avg={test_out['baseline_avg']}")
print(f"  dates:  {test_out['dates']}")
print(f"  values: {test_out['values']}")
print(f"  events: {[e for e in test_out['events'] if e]}")

os.makedirs("src/models", exist_ok=True)
joblib.dump(demand_model, "src/models/demand_model.pkl")
print("Saved → src/models/demand_model.pkl")
