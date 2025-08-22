# data_prep/synth_history.py
from __future__ import annotations
import pandas as pd
import numpy as np
from datetime import date, timedelta

def gen_daily_range(days: int = 180):
    today = date.today()
    start = today - timedelta(days=days)
    return pd.date_range(start, periods=days, freq="D")

def is_spot_candidate(row: pd.Series) -> bool:
    txt = f"{row['service']} {row['sku']}".lower()
    return any(k in txt for k in ["gpu", "compute", "ec2", "vm"])

def synth_history_from_snapshot(snapshot_csv: str, out_csv: str, days: int = 180, seed: int = 7):
    rng = np.random.default_rng(seed)
    snap = pd.read_csv(snapshot_csv)
    dates = gen_daily_range(days)

    rows = []
    for _, r in snap.iterrows():
        base = float(r["unit_price"])
        # small drift for storage/egress; higher variance for compute/gpu (spot‑like)
        if is_spot_candidate(r):
            # mean ~ 60% of on‑demand with noise
            mean = base * 0.6
            noise = rng.normal(0, mean*0.08, size=len(dates))
            series = np.clip(mean + noise, 0.01, None)
        else:
            # stable series with tiny drift
            drift = rng.normal(0, base*0.002, size=len(dates)).cumsum()
            series = np.clip(base + drift, 0.0, None)

        for d, p in zip(dates, series):
            rows.append({
                "date": d.date().isoformat(),
                "cloud": r["cloud"],
                "region": r["region"],
                "service": r["service"],
                "sku": r["sku"],
                "unit": r["unit"],
                "unit_price": round(float(p), 6),
            })

    out = pd.DataFrame(rows)
    out.to_csv(out_csv, index=False)
    return out