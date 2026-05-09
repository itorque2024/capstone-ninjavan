"""
Generate enriched training datasets for all 3 ML models.

Run:
    conda run -n ninjavan python src/scripts/generate_datasets.py

Outputs:
    ninjavan_optionB_datasets/fraud_data.csv        — 10 000 rows, 6 features, 5% fraud
    ninjavan_optionB_datasets/maintenance_data.csv  — 10 000 rows, 6 features + label
    ninjavan_optionB_datasets/demand_data.csv       — ~15 000 rows, 6 SEA regions, 2+ yrs + holidays
"""
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
OUT = _ROOT / "ninjavan_optionB_datasets"
OUT.mkdir(exist_ok=True)


# ── 1. Fraud Detection Dataset ─────────────────────────────────────────────────

def generate_fraud_data(n_total: int = 10_000, fraud_rate: float = 0.05) -> pd.DataFrame:
    n_fraud = int(n_total * fraud_rate)      # 500
    n_legit = n_total - n_fraud              # 9 500

    # ── Legitimate claims (95%) ──────────────────────────────────────────────
    legit_value        = np.exp(RNG.normal(3.5, 0.8, n_legit)).clip(5, 200)
    legit_prior        = RNG.choice([0, 0, 0, 0, 0, 1, 1, 1, 2], size=n_legit)
    legit_age          = RNG.integers(90, 1800, size=n_legit)
    legit_lag          = RNG.integers(5, 30, size=n_legit)

    legit_df = pd.DataFrame({
        "parcel_value":    legit_value.round(2),
        "prior_claims":    legit_prior,
        "account_age_days": legit_age,
        "claim_lag_days":  legit_lag,
        "fraud_flag":      0,
    })

    # ── Fraudulent claims — 3 behavioural patterns ──────────────────────────
    # Pattern 1 (40%): high-value parcel + very new account + fast claim
    n1 = int(n_fraud * 0.40)
    p1_value = RNG.uniform(150, 500, n1).round(2)
    p1_prior = RNG.choice([0, 0, 1], size=n1)
    p1_age   = RNG.integers(1, 30, size=n1)
    p1_lag   = RNG.integers(0, 5, size=n1)

    # Pattern 2 (40%): repeat claimers — many prior claims, quick filing
    n2 = int(n_fraud * 0.40)
    p2_value = np.exp(RNG.normal(3.8, 0.6, n2)).clip(20, 400).round(2)
    p2_prior = RNG.integers(4, 9, size=n2)
    p2_age   = RNG.integers(20, 365, size=n2)
    p2_lag   = RNG.integers(0, 3, size=n2)

    # Pattern 3 (20%): very high value + brand-new account + multiple prior claims
    n3 = n_fraud - n1 - n2
    p3_value = RNG.uniform(200, 500, n3).round(2)
    p3_prior = RNG.integers(3, 7, size=n3)
    p3_age   = RNG.integers(1, 20, size=n3)
    p3_lag   = RNG.integers(0, 7, size=n3)

    fraud_parts = []
    for val, pri, age, lag, ni in [
        (p1_value, p1_prior, p1_age, p1_lag, n1),
        (p2_value, p2_prior, p2_age, p2_lag, n2),
        (p3_value, p3_prior, p3_age, p3_lag, n3),
    ]:
        fraud_parts.append(pd.DataFrame({
            "parcel_value":    val,
            "prior_claims":    pri,
            "account_age_days": age,
            "claim_lag_days":  lag,
            "fraud_flag":      1,
        }))

    fraud_df = pd.concat(fraud_parts, ignore_index=True)

    df = pd.concat([legit_df, fraud_df], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df.insert(0, "parcel_id", range(1, len(df) + 1))
    return df


fraud_df = generate_fraud_data()
fraud_df.to_csv(OUT / "fraud_data.csv", index=False)
print(f"fraud_data.csv — shape: {fraud_df.shape}")
print(f"  Fraud rate: {fraud_df['fraud_flag'].mean()*100:.2f}%")
print(f"  Columns: {list(fraud_df.columns)}")
print(f"  parcel_value: {fraud_df['parcel_value'].describe()[['min','mean','max']].round(1).to_dict()}")


# ── 2. Predictive Maintenance Dataset ─────────────────────────────────────────

def generate_maintenance_data(n_total: int = 10_000) -> pd.DataFrame:
    vehicle_ids = RNG.integers(200, 300, size=n_total)  # vehicles 200–299

    # Health score: bimodal — mostly healthy (70–95), some at risk (30–65)
    healthy_mask = RNG.random(n_total) > 0.25
    health = np.where(
        healthy_mask,
        RNG.normal(80, 10, n_total).clip(60, 100),
        RNG.normal(45, 12, n_total).clip(10, 65),
    )

    # Sensor readings — physically correlated with health score
    engine_temp  = (90 + (100 - health) * 0.85 + RNG.normal(0, 4, n_total)).clip(80, 210)
    tyre_pressure = (225 + (health - 50) * 0.55 + RNG.normal(0, 6, n_total)).clip(145, 265)
    vibration    = (0.08 + (100 - health) * 0.009 + RNG.exponential(0.04, n_total)).clip(0, 2.5)
    km_service   = ((100 - health) * 85 + RNG.uniform(0, 1200, n_total)).clip(0, 15000)

    at_risk = (health < 60).astype(int)

    df = pd.DataFrame({
        "vehicle_id":         vehicle_ids,
        "vehicle_health_score": health.round(2),
        "engine_temp_c":      engine_temp.round(1),
        "tyre_pressure_kpa":  tyre_pressure.round(1),
        "vibration_g":        vibration.round(3),
        "km_since_service":   km_service.round(0).astype(int),
        "at_risk":            at_risk,
    })
    return df


maint_df = generate_maintenance_data()
maint_df.to_csv(OUT / "maintenance_data.csv", index=False)
print(f"\nmaintenance_data.csv — shape: {maint_df.shape}")
print(f"  At-risk rate: {maint_df['at_risk'].mean()*100:.1f}%")
print(f"  Columns: {list(maint_df.columns)}")


# ── 3. Demand Forecasting Dataset — 6 SEA regions + holidays/promos ───────────

from src.utils.sea_holidays import PROMO_DAYS, get_region_events


def _build_multiplier_map(dates: pd.DatetimeIndex, region: str) -> np.ndarray:
    """
    Return a per-day multiplier array for a single region.
    Layers: promo days → region holidays (additive on top of base 1.0).
    """
    mult = np.ones(len(dates))
    date_index = {d.date(): i for i, d in enumerate(dates)}

    # ── E-commerce promotion days (all regions) ─────────────────────────────
    for year in range(2022, 2025):
        for (month, day), (name, day_mult, pre_days, pre_mult) in PROMO_DAYS.items():
            try:
                event_date = pd.Timestamp(year=year, month=month, day=day).date()
            except ValueError:
                continue
            # Day itself
            if event_date in date_index:
                mult[date_index[event_date]] = max(mult[date_index[event_date]], day_mult)
            # Pre-event ramp (linear ramp from 1.0 → pre_mult)
            for d in range(1, pre_days + 1):
                ramp_date = (pd.Timestamp(event_date) - pd.Timedelta(days=d)).date()
                ramp_mult = 1.0 + (pre_mult - 1.0) * (pre_days - d + 1) / pre_days
                if ramp_date in date_index:
                    mult[date_index[ramp_date]] = max(mult[date_index[ramp_date]], ramp_mult)

    # ── Region-specific holidays ─────────────────────────────────────────────
    for (date_str, name, regions, pre_days, pre_mult, peak_mult, dip_days, dip_mult, post_days, post_mult) in get_region_events():
        if region not in regions:
            continue
        event_date = pd.Timestamp(date_str).date()

        # Pre-event ramp
        for d in range(1, pre_days + 1):
            ramp_date = (pd.Timestamp(event_date) - pd.Timedelta(days=d)).date()
            ramp_mult = 1.0 + (pre_mult - 1.0) * (pre_days - d + 1) / pre_days
            if ramp_date in date_index:
                mult[date_index[ramp_date]] = max(mult[date_index[ramp_date]], ramp_mult)

        # Holiday dip (delivery shutdown)
        for d in range(dip_days + 1):
            dip_date = (pd.Timestamp(event_date) + pd.Timedelta(days=d)).date()
            if dip_date in date_index:
                mult[date_index[dip_date]] = dip_mult  # override — always applies

        # Post-holiday bounce
        for d in range(1, post_days + 1):
            bounce_date = (pd.Timestamp(event_date) + pd.Timedelta(days=dip_days + d)).date()
            if bounce_date in date_index:
                mult[date_index[bounce_date]] = max(mult[date_index[bounce_date]], post_mult)

    return mult


def generate_demand_data() -> pd.DataFrame:
    dates = pd.date_range("2022-01-01", "2024-12-31", freq="D")

    region_params = {
        # region: (base_vol, weekly_amp, yearly_amp, growth_rate, noise_std)
        "ID": (700, 60, 80,  0.00030, 30),
        "MY": (480, 40, 55,  0.00025, 25),
        "SG": (350, 30, 40,  0.00020, 20),
        "PH": (420, 50, 60,  0.00028, 28),
        "TH": (310, 35, 45,  0.00022, 22),
        "VN": (260, 30, 35,  0.00018, 18),
    }

    rows = []
    t = np.arange(len(dates))

    for region, (base, w_amp, y_amp, growth, noise_std) in region_params.items():
        trend  = base * np.exp(growth * t)
        weekly = w_amp * np.sin(2 * np.pi * t / 7)
        yearly = y_amp * np.sin(2 * np.pi * t / 365.25 - np.pi / 2)
        noise  = RNG.normal(0, noise_std, len(t))
        volume = (trend + weekly + yearly + noise).clip(10)

        # Apply holiday + promotion multipliers
        mult = _build_multiplier_map(dates, region)
        volume = (volume * mult).clip(10).round().astype(int)

        for date, vol in zip(dates, volume):
            rows.append({"order_date": date.strftime("%Y-%m-%d"), "region": region, "parcel_volume": int(vol)})

    df = pd.DataFrame(rows)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df


demand_df = generate_demand_data()
demand_df.to_csv(OUT / "demand_data.csv", index=False)
print(f"\ndemand_data.csv — shape: {demand_df.shape}")
print(f"  Regions: {sorted(demand_df['region'].unique())}")
print(f"  Date range: {demand_df['order_date'].min()} → {demand_df['order_date'].max()}")
print(f"  Volume per region (mean):")
for r, v in demand_df.groupby("region")["parcel_volume"].mean().sort_values(ascending=False).items():
    print(f"    {r}: {v:.0f} parcels/day")

print("\nAll datasets generated successfully.")
