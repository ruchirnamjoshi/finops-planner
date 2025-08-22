from __future__ import annotations
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRICING_DIR = ROOT / "data" / "pricing"

def load_history_csv() -> pd.DataFrame | None:
    """Load the synthetic SKU history for ForecastBot."""
    path = PRICING_DIR / "sku_history.csv"
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        # normalize schema
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        return df
    except Exception as e:
        print(f"[WARN] could not load sku_history.csv: {e}")
        return None