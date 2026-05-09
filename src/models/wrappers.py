"""
Pickle-safe wrapper classes for the three trained models.
Defined here (not in notebooks/__main__) so joblib can find them at load time.
"""
import numpy as np
import pandas as pd


class DemandForecastModel:
    """
    Wrapper used by demand_agent.py.
    predict(horizon, start_date, region) -> rich dict with dates, CI, events, fleet sizing.
    """

    def __init__(self, prophet_model, scaler, history_values, seq_len,
                 lstm_model=None, prophet_weight=0.6,
                 training_end_date=None, region_shares=None):
        self.prophet = prophet_model
        self.scaler = scaler
        self.history_values = np.array(history_values)
        self.seq_len = seq_len
        self.prophet_weight = prophet_weight
        self.baseline_avg = float(np.mean(history_values))
        self._lstm = lstm_model
        # Last date in training data (so we know how far ahead to forecast)
        self.training_end_date = pd.Timestamp(training_end_date) if training_end_date else pd.Timestamp("2024-12-31")
        # Historical parcel share per region (e.g. {"ID": 0.29, "MY": 0.19, ...})
        self.region_shares = region_shares or {}

    # ── Event label lookup ────────────────────────────────────────────────────
    @staticmethod
    def _event_map(start: pd.Timestamp, end: pd.Timestamp) -> dict:
        """Return {date: label} for all holidays/promos in [start, end]."""
        try:
            from src.utils.sea_holidays import PROMO_DAYS, get_region_events
        except ImportError:
            return {}

        events = {}
        promo_labels = {
            "sale_9_9": "🛍 9.9 Sale",
            "sale_10_10": "🛍 10.10 Sale",
            "sale_11_11": "🛍 11.11 Singles' Day",
            "sale_12_12": "🛍 12.12 Sale",
        }
        for year in range(start.year, end.year + 1):
            for (month, day), (name, *_) in PROMO_DAYS.items():
                try:
                    dt = pd.Timestamp(year=year, month=month, day=day)
                except ValueError:
                    continue
                if start <= dt <= end:
                    events[dt.date()] = promo_labels.get(name, name)

        hol_labels = {
            "eid_al_fitr":      "🕌 Eid al-Fitr (ID/MY)",
            "chinese_new_year": "🧧 Chinese New Year (SG/MY)",
            "tet":              "🎆 Tết (VN)",
            "songkran":         "💦 Songkran (TH)",
            "christmas_ph":     "🎄 Christmas (PH)",
            "diwali":           "🪔 Diwali (SG/MY)",
            "new_year":         "🎉 New Year",
        }
        for (date_str, name, regions, pre_days, pre_m, peak_m, dip_days, dip_m, *_) in get_region_events():
            dt = pd.Timestamp(date_str)
            if start <= dt <= end and name in hol_labels:
                events[dt.date()] = hol_labels[name]

        return events

    def predict(self, horizon: int = 14, start_date: str = None, warehouse: str = "All", marketing_spend: float = 0.0, is_sale: bool = False) -> dict:
        """
        Forecast `horizon` days starting from `start_date` (default: tomorrow).
        Returns dates, point forecast, 95% CI, event labels, fleet sizing, warehouse breakdown.
        """
        import math

        start = pd.Timestamp(start_date).normalize() if start_date else (
            pd.Timestamp.today().normalize() + pd.Timedelta(days=1)
        )
        end = start + pd.Timedelta(days=horizon - 1)

        # How many periods to request from Prophet (from training end → end)
        periods_needed = max(int((end - self.training_end_date).days) + 5, horizon + 5)
        future = self.prophet.make_future_dataframe(periods=periods_needed)
        fc = self.prophet.predict(future)

        # Slice to requested window
        mask = (fc["ds"] >= start) & (fc["ds"] <= end)
        fc_slice = fc[mask].reset_index(drop=True)

        dates  = fc_slice["ds"].dt.strftime("%Y-%m-%d").tolist()
        values = fc_slice["yhat"].clip(lower=0).round().astype(int).tolist()
        lower  = fc_slice["yhat_lower"].clip(lower=0).round().astype(int).tolist()
        upper  = fc_slice["yhat_upper"].clip(lower=0).round().astype(int).tolist()

        # Event annotations
        event_map = self._event_map(start, end)
        events = [event_map.get(pd.Timestamp(d).date(), "") for d in dates]

        # Simulated Feature Engineering (LSTM Exogenous Variables)
        # Marketing spend gradually lifts the baseline.
        marketing_multiplier = 1.0 + (marketing_spend / 10000.0) * 0.03  # 3% boost per $10k

        for i in range(len(values)):
            values[i] = int(values[i] * marketing_multiplier)
            lower[i] = int(lower[i] * marketing_multiplier)
            upper[i] = int(upper[i] * marketing_multiplier)

            # Massive spike on the first day if there is a major e-commerce sale
            if is_sale and i == 0:
                values[i] = int(values[i] * 2.5)
                upper[i] = int(upper[i] * 2.5)
                events[i] = "🛒 MAJOR E-COMMERCE SALE"
            # Lingering effect on day 2
            elif is_sale and i == 1:
                values[i] = int(values[i] * 1.5)
                upper[i] = int(upper[i] * 1.5)
                events[i] = "📦 SALE AFTERMATH"

        # Per-warehouse breakdown
        warehouse_shares = {
            "Tampines Hub": 0.40,
            "Jurong Hub": 0.35,
            "Changi Hub": 0.15,
            "Woodlands Hub": 0.10
        }

        warehouse_breakdown = {}
        if warehouse == "All":
            for w, share in warehouse_shares.items():
                warehouse_breakdown[w] = [int(v * share) for v in values]
        elif warehouse in warehouse_shares:
            share = warehouse_shares[warehouse]
            values = [int(v * share) for v in values]
            lower  = [int(v * share) for v in lower]
            upper  = [int(v * share) for v in upper]
            warehouse_breakdown = {warehouse: values}

        # Fleet sizing: 80 parcels / rider / day
        riders = [math.ceil(v / 80) for v in values]

        # Spike flags
        baseline = float(np.mean(self.history_values))
        if warehouse != "All" and warehouse in warehouse_shares:
            baseline = baseline * warehouse_shares[warehouse]

        return {
            "baseline_avg":     round(baseline),
            "values":           values,
            "values_lower":     lower,
            "values_upper":     upper,
            "dates":            dates,
            "events":           events,
            "riders_needed":    riders,
            "warehouse_breakdown": warehouse_breakdown,
            "horizon":          horizon,
            "start_date":       dates[0] if dates else str(start.date()),
        }


class MaintenanceRiskModel:
    """Wrapper used by maintenance_agent.py. Imputes missing sensor columns from health score."""

    def __init__(self, xgb_model, feature_cols):
        self.model = xgb_model
        self.feature_cols = feature_cols

    def _prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        hs = out.get('vehicle_health_score', pd.Series([70.0] * len(out)))
        if 'engine_temp_c' not in out.columns:
            out['engine_temp_c'] = (90 + (100 - hs) * 0.8).clip(80, 200)
        if 'tyre_pressure_kpa' not in out.columns:
            out['tyre_pressure_kpa'] = (220 + (hs - 50) * 0.5).clip(150, 260)
        if 'vibration_g' not in out.columns:
            out['vibration_g'] = ((100 - hs) * 0.008 + 0.1).clip(0, 2)
        if 'km_since_service' not in out.columns:
            out['km_since_service'] = ((100 - hs) * 80).clip(0, 15000)
        if 'health_band' not in out.columns:
            out['health_band'] = pd.cut(hs, bins=[0, 40, 60, 80, 100], labels=[3, 2, 1, 0]).astype(int)
        return out[self.feature_cols].fillna(0)

    def predict_proba(self, df: pd.DataFrame):
        return self.model.predict_proba(self._prepare(df))

    def predict(self, df: pd.DataFrame):
        return self.model.predict(self._prepare(df))


class FraudDetectionModel:
    """Wrapper used by fraud_agent.py. Imputes features from parcel_value alone."""

    def __init__(self, iso_forest, lgbm_model, feature_cols, train_stats, threshold):
        self.iso = iso_forest
        self.lgbm = lgbm_model
        self.feature_cols = feature_cols
        self.stats = train_stats
        self.threshold = threshold

    def _prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        pv = out.get('parcel_value', pd.Series([self.stats['value_mean']] * len(out)))
        out['value_log']     = np.log1p(pv)
        out['value_zscore']  = (pv - self.stats['value_mean']) / (self.stats['value_std'] + 1e-9)
        out['value_pct']     = pv.rank(pct=True) if len(pv) > 1 else pv / max(float(pv.max()), 1.0)
        out['is_high_value'] = (pv > self.stats['p95']).astype(int)
        out['is_very_low']   = (pv < self.stats['p02']).astype(int)
        for col in ['prior_claims', 'account_age_days', 'claim_lag_days']:
            if col not in out.columns:
                out[col] = {'prior_claims': 0, 'account_age_days': 365, 'claim_lag_days': 7}[col]
        return out[self.feature_cols].fillna(0)

    def decision_function(self, df: pd.DataFrame):
        return self.iso.decision_function(self._prepare(df))

    def risk_probability(self, df: pd.DataFrame):
        """LightGBM continuous fraud probability [0, 1] — used for scoring and ranking."""
        return self.lgbm.predict_proba(self._prepare(df))[:, 1]

    def predict_fraud(self, df: pd.DataFrame):
        prob = self.lgbm.predict_proba(self._prepare(df))[:, 1]
        return (prob >= self.threshold).astype(int)
