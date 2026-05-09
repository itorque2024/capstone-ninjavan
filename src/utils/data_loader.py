import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent.parent / "ninjavan_optionB_datasets"


def load_demand_data() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "demand_data.csv", parse_dates=["order_date"])


def load_maintenance_data() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "maintenance_data.csv")


def load_fraud_data() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "fraud_data.csv")


def load_master_data() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "ninjavan_synthetic_master.csv", parse_dates=["order_date", "delivery_date"])
