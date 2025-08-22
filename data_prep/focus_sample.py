# data_prep/focus_sample.py
from __future__ import annotations
import pandas as pd
from datetime import date, timedelta

def make_focus_like_sample(out_csv: str, days: int = 90):
    start = date.today() - timedelta(days=days)
    dates = pd.date_range(start, periods=days, freq="D")
    rows = []
    for d in dates:
        # simple synthetic usage & cost for demo
        rows += [
            {"date": d.date().isoformat(), "provider": "gcp", "account": "demo",
             "service": "compute_engine", "sku": "a2-highgpu-1g",
             "usage_qty": 0.08, "usage_unit": "instance_day", "cost": 110.0},
            {"date": d.date().isoformat(), "provider": "gcp", "account": "demo",
             "service": "gcs", "sku": "standard_storage",
             "usage_qty": 80.0, "usage_unit": "gb_day", "cost": 1.84},
            {"date": d.date().isoformat(), "provider": "gcp", "account": "demo",
             "service": "network", "sku": "egress",
             "usage_qty": 15.0, "usage_unit": "gb_day", "cost": 1.8},
        ]
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    return df