# data_prep/build_data.py
from __future__ import annotations
import os, duckdb, pandas as pd
from pathlib import Path
from .synth_history import synth_history_from_snapshot
from .focus_sample import make_focus_like_sample
ROOT = Path(__file__).resolve().parents[1]
PRICING_DIR = ROOT / "data" / "pricing"
FOCUS_DIR = ROOT / "data" / "focus"
DB_PATH = ROOT / "data" / "finops.duckdb"

def ensure_dirs():
    PRICING_DIR.mkdir(parents=True, exist_ok=True)
    FOCUS_DIR.mkdir(parents=True, exist_ok=True)

def write_egress_matrix():
    """Rough egress matrix for demo (per‑GB). Adjust later from real sources."""
    df = pd.DataFrame([
        # same‑region cheap
        {"cloud": "gcp", "src_region": "us-central1", "dst_region": "us-central1", "price_per_gb": 0.01},
        # cross‑region typical
        {"cloud": "gcp", "src_region": "us-central1", "dst_region": "us-east1", "price_per_gb": 0.12},
        {"cloud": "aws", "src_region": "us-east-1", "dst_region": "us-west-2", "price_per_gb": 0.09},
        {"cloud": "azure", "src_region": "eastus", "dst_region": "westus2", "price_per_gb": 0.087},
    ])
    out = PRICING_DIR / "egress_matrix.csv"
    df.to_csv(out, index=False)
    return out

def build_duckdb():
    # try to open; if it's invalid, delete and recreate
    try:
        con = duckdb.connect(DB_PATH.as_posix())
    except Exception:
        # remove corrupted/invalid file and recreate
        if DB_PATH.exists():
            DB_PATH.unlink()
        con = duckdb.connect(DB_PATH.as_posix())

    # pricing snapshot
    snap = pd.read_csv(PRICING_DIR / "sku_snapshot.csv")
    con.execute("CREATE OR REPLACE TABLE sku_snapshot AS SELECT * FROM snap")

    # pricing history
    hist = pd.read_csv(PRICING_DIR / "sku_history.csv")
    con.execute("CREATE OR REPLACE TABLE sku_history AS SELECT * FROM hist")

    # egress matrix
    egress = pd.read_csv(PRICING_DIR / "egress_matrix.csv")
    con.execute("CREATE OR REPLACE TABLE egress_matrix AS SELECT * FROM egress")

    # focus cost/usage sample
    cus = pd.read_csv(FOCUS_DIR / "cost_usage_sample.csv")
    con.execute("CREATE OR REPLACE TABLE cost_usage_sample AS SELECT * FROM cus")

    con.close()

def main():
    ensure_dirs()
    # 1) generate history from snapshot
    history_csv = PRICING_DIR / "sku_history.csv"
    synth_history_from_snapshot(
        snapshot_csv=(PRICING_DIR / "sku_snapshot.csv").as_posix(),
        out_csv=history_csv.as_posix(),
        days=180,
        seed=11,
    )
    # 2) egress matrix
    write_egress_matrix()
    # 3) FOCUS-like sample
    make_focus_like_sample((FOCUS_DIR / "cost_usage_sample.csv").as_posix(), days=90)
    # 4) cache in duckdb
    build_duckdb()
    print(f"Built: {DB_PATH}")

if __name__ == "__main__":
    main()