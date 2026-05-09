"""
SEA public holidays and e-commerce promotion days (2022-2024).

Two outputs:
  get_prophet_holidays()   → pd.DataFrame for Prophet's holidays parameter
  REGION_EVENTS            → dict used by demand data generator for per-region spike multipliers
"""
import pandas as pd

# ── E-commerce mega-sale days (all 6 regions) ─────────────────────────────────
PROMO_DAYS = {
    # (month, day): (name, day_multiplier, pre_days, pre_multiplier)
    (9,  9):  ("sale_9_9",   3.0, 4, 1.6),
    (10, 10): ("sale_10_10", 3.5, 4, 1.7),
    (11, 11): ("sale_11_11", 7.0, 10, 2.2),   # Singles' Day — biggest SEA event
    (12, 12): ("sale_12_12", 4.5, 6, 1.8),
}

# ── Region-specific public holidays ───────────────────────────────────────────
# Each entry: (date_str, name, affected_regions, pre_days, pre_mult, peak_mult, dip_days, dip_mult, post_days, post_mult)
# dip_days: days where delivery actually slows (holiday shutdown)
_RAW_EVENTS = [
    # ── Eid al-Fitr (Lebaran) — ID + MY ──────────────────────────────────────
    ("2022-05-02", "eid_al_fitr", ["ID", "MY"], 14, 2.8, 0.0, 3, 0.15, 7, 1.6),
    ("2023-04-21", "eid_al_fitr", ["ID", "MY"], 14, 2.8, 0.0, 3, 0.15, 7, 1.6),
    ("2024-04-10", "eid_al_fitr", ["ID", "MY"], 14, 2.8, 0.0, 3, 0.15, 7, 1.6),

    # ── Chinese New Year — SG + MY + VN ──────────────────────────────────────
    ("2022-02-01", "chinese_new_year", ["SG", "MY", "VN"], 12, 2.4, 0.0, 3, 0.2, 5, 1.4),
    ("2023-01-22", "chinese_new_year", ["SG", "MY", "VN"], 12, 2.4, 0.0, 3, 0.2, 5, 1.4),
    ("2024-02-10", "chinese_new_year", ["SG", "MY", "VN"], 12, 2.4, 0.0, 3, 0.2, 5, 1.4),

    # ── Tết (Lunar New Year) — VN only (larger effect) ────────────────────────
    ("2022-02-01", "tet", ["VN"], 18, 3.2, 0.0, 5, 0.1, 7, 1.7),
    ("2023-01-22", "tet", ["VN"], 18, 3.2, 0.0, 5, 0.1, 7, 1.7),
    ("2024-02-10", "tet", ["VN"], 18, 3.2, 0.0, 5, 0.1, 7, 1.7),

    # ── Songkran — TH ────────────────────────────────────────────────────────
    ("2022-04-13", "songkran", ["TH"], 5, 1.8, 0.0, 4, 0.3, 3, 1.3),
    ("2023-04-13", "songkran", ["TH"], 5, 1.8, 0.0, 4, 0.3, 3, 1.3),
    ("2024-04-13", "songkran", ["TH"], 5, 1.8, 0.0, 4, 0.3, 3, 1.3),

    # ── Christmas + Merry December — PH (sustained build-up from Sep) ─────────
    ("2022-12-25", "christmas_ph", ["PH"], 30, 2.5, 0.0, 2, 0.4, 5, 1.5),
    ("2023-12-25", "christmas_ph", ["PH"], 30, 2.5, 0.0, 2, 0.4, 5, 1.5),
    ("2024-12-25", "christmas_ph", ["PH"], 30, 2.5, 0.0, 2, 0.4, 5, 1.5),

    # ── Deepavali/Diwali — SG + MY ────────────────────────────────────────────
    ("2022-10-24", "diwali", ["SG", "MY"], 7, 1.8, 1.5, 1, 0.6, 3, 1.2),
    ("2023-11-12", "diwali", ["SG", "MY"], 7, 1.8, 1.5, 1, 0.6, 3, 1.2),
    ("2024-10-31", "diwali", ["SG", "MY"], 7, 1.8, 1.5, 1, 0.6, 3, 1.2),

    # ── Philippine Independence + New Year ────────────────────────────────────
    ("2022-01-01", "new_year", ["PH", "TH", "VN"], 5, 1.6, 0.0, 2, 0.5, 3, 1.2),
    ("2023-01-01", "new_year", ["PH", "TH", "VN"], 5, 1.6, 0.0, 2, 0.5, 3, 1.2),
    ("2024-01-01", "new_year", ["PH", "TH", "VN"], 5, 1.6, 0.0, 2, 0.5, 3, 1.2),
]


def get_region_events() -> list:
    """Return raw event list for demand data generator."""
    return _RAW_EVENTS


def get_prophet_holidays() -> pd.DataFrame:
    """
    Build a Prophet-compatible holidays DataFrame covering all SEA events
    and promotion days (2022-2024 + 2025 for forecast window).
    """
    rows = []

    # ── Promotion days (every year 2022-2025) ─────────────────────────────────
    for year in range(2022, 2026):
        for (month, day), (name, day_mult, pre_days, _) in PROMO_DAYS.items():
            rows.append({
                "holiday": name,
                "ds": pd.Timestamp(year=year, month=month, day=day),
                "lower_window": -pre_days,
                "upper_window": 1,
            })

    # ── Public holidays ────────────────────────────────────────────────────────
    seen = set()
    for (date_str, name, regions, pre_days, _, _, dip_days, _, post_days, _) in _RAW_EVENTS:
        key = (date_str, name)
        if key in seen:
            continue
        seen.add(key)
        rows.append({
            "holiday": name,
            "ds": pd.Timestamp(date_str),
            "lower_window": -pre_days,
            "upper_window": post_days,
        })

    return pd.DataFrame(rows).drop_duplicates(subset=["holiday", "ds"]).reset_index(drop=True)
