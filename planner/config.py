# planner/config.py
from __future__ import annotations
import os, pathlib, yaml

DEFAULT_PATH = pathlib.Path(__file__).resolve().parents[1] / "settings.yaml"

class Settings(dict):
    @property
    def blueprints(self): return self["blueprints"]
    @property
    def db_path(self):    return self["data"]["db_path"]
    @property
    def history_csv(self): return self["data"]["history_csv"]
    @property
    def price_snapshot(self): return self["data"]["price_snapshot"]
    @property
    def forecast_model(self): return self["forecast"].get("model", "sarima")
    @property
    def horizon_days(self):   return int(self["forecast"].get("horizon_days", 30))

def _resolve(p, base):
    p = pathlib.Path(p)
    return (base / p if not p.is_absolute() else p).resolve()

def load_settings(path: str | os.PathLike | None = None) -> Settings:
    path = pathlib.Path(os.environ.get("FINOPS_SETTINGS", path or DEFAULT_PATH)).resolve()
    if not path.exists():
        raise FileNotFoundError(f"settings.yaml not found at {path}")
    with open(path, "r") as f:
        raw = yaml.safe_load(f) or {}
    base = path.parent

    # Basic validation + path normalization
    required = ["data", "blueprints", "forecast"]
    for k in required:
        if k not in raw:
            raise ValueError(f"settings.yaml missing top-level key: {k}")

    raw["data"]["db_path"]       = str(_resolve(raw["data"]["db_path"], base))
    raw["data"]["history_csv"]   = str(_resolve(raw["data"]["history_csv"], base))
    raw["data"]["price_snapshot"]= str(_resolve(raw["data"]["price_snapshot"], base))
    raw["blueprints"] = [str(_resolve(p, base)) for p in raw["blueprints"]]

    return Settings(raw)