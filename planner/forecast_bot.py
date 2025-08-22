from __future__ import annotations
from typing import List, Tuple, Dict
import pandas as pd
import numpy as np
from datetime import timedelta

from pmdarima import auto_arima
from statsmodels.tsa.holtwinters import ExponentialSmoothing

import warnings

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    _TF_AVAILABLE = True
except Exception:
    _TF_AVAILABLE = False

# ---- Helpers ---------------------------------------------------------------

def _scale_minmax(a: np.ndarray) -> tuple[np.ndarray, float, float]:
    amin = float(np.min(a)) if len(a) else 0.0
    amax = float(np.max(a)) if len(a) else 1.0
    if not np.isfinite(amin) or not np.isfinite(amax) or (amax - amin) < 1e-9:
        # flat series → return zeros with safe bounds
        return np.zeros_like(a, dtype=float), 0.0, 1.0
    return (a - amin) / (amax - amin), amin, amax

def _unscale_minmax(a_scaled: np.ndarray, amin: float, amax: float) -> np.ndarray:
    return a_scaled * (amax - amin) + amin

def _make_supervised(y: np.ndarray, lookback: int) -> tuple[np.ndarray, np.ndarray]:
    X, T = [], []
    for i in range(lookback, len(y)):
        X.append(y[i - lookback:i])
        T.append(y[i])
    X = np.asarray(X, dtype=float).reshape(-1, lookback, 1)
    T = np.asarray(T, dtype=float)
    return X, T

def _per_day_qty(unit: str, monthly_qty: float) -> float:
    """
    Convert a monthly quantity to a daily quantity; for pure daily units just return qty.
    We assume BoM quantities are "monthly" for *_month and total monthly for egress gb.
    """
    u = (unit or "").lower()
    if u in ("instance_month", "gb_month", "gb"):  # egress gb per month
        return float(monthly_qty) / 30.0
    elif u in ("gb_day", "instance_day"):
        return float(monthly_qty)
    else:
        return float(monthly_qty) / 30.0


def _ensure_daily(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse to unique (date, sku) rows and ensure a continuous daily index
    after pivoting (date x sku). We ffill missing days per SKU.
    """
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])
    # keep last observation per (date, sku)
    d = d.sort_values(["sku", "date"]).groupby(["date", "sku"], as_index=False).last()

    # pivot: rows=date, cols=sku
    pivot = d.pivot(index="date", columns="sku", values="unit_price").sort_index()

    # build full daily index and forward-fill
    all_days = pd.date_range(pivot.index.min(), pivot.index.max(), freq="D")
    pivot = pivot.reindex(all_days).ffill()
    pivot.index.name = "date"
    return pivot  # <- returns wide (date x sku)

def _fit_forecaster(series: pd.Series, seasonal: bool = False, model_name: str = "auto"):
    y = series.astype(float).values
    n = len(y)

    # Guard rails
    if n < 10 or np.allclose(np.var(y), 0.0):
        level = float(np.mean(y)) if n else 0.0
        return lambda h: np.full(h, level, dtype=float)

    m = 7 if seasonal and n >= 42 else 1  # weekly seasonality only when long enough

    # Explicit selections
    name = (model_name or "auto").lower()

    if name == "lstm":
        try:
            return _fit_lstm_forecaster(series, seasonal=seasonal)
        except Exception as e:
            warnings.warn(f"LSTM failed ({e}); falling back to ARIMA/ETS.")

    if name == "arima":
        try:
            model = auto_arima(
                y, start_p=0, start_q=0, max_p=3, max_q=3,
                start_P=0, start_Q=0, max_P=2, max_Q=2, seasonal=(m > 1), m=m,
                stepwise=True, suppress_warnings=True, error_action="ignore", maxiter=100
            )
            return lambda h: model.predict(n_periods=h)
        except Exception:
            pass

    if name == "ets":
        try:
            ets = ExponentialSmoothing(
                series, trend="add",
                seasonal="add" if m > 1 else None,
                seasonal_periods=m if m > 1 else None,
                initialization_method="estimated"
            ).fit()
            return lambda h: ets.forecast(h).values
        except Exception:
            pass

    # Auto: ARIMA → ETS → mean
    try:
        model = auto_arima(y, seasonal=(m > 1), m=m, stepwise=True, suppress_warnings=True)
        return lambda h: model.predict(n_periods=h)
    except Exception:
        try:
            ets = ExponentialSmoothing(series, trend="add", initialization_method="estimated").fit()
            return lambda h: ets.forecast(h).values
        except Exception:
            level = float(np.mean(y))
            return lambda h: np.full(h, level, dtype=float)

def _fit_lstm_forecaster(series: pd.Series, seasonal: bool = False) -> callable:
    """
    Train a tiny LSTM on a single univariate price series.
    Returns a callable f(h) -> np.ndarray of length h.
    """
    if not _TF_AVAILABLE:
        warnings.warn("TensorFlow not available; LSTM forecasting not possible.")
        # No fallbacks - raise error
        raise RuntimeError("TensorFlow required for LSTM forecasting - no fallback available")

    y = series.astype(float).values
    n = len(y)

    # Guards: if too short or flat, no fallback
    if n < 30 or np.allclose(np.var(y), 0.0, atol=1e-12):
        raise RuntimeError(f"Series too short ({n} points) or flat for LSTM forecasting")

    # Lookback: weekly window if seasonal and enough history, else 28 default
    lookback = 42 if (seasonal and n >= 84) else 28
    lookback = min(max(lookback, 10), n - 1)  # clamp

    # Scale and build supervised dataset
    y_scaled, amin, amax = _scale_minmax(y)
    X, T = _make_supervised(y_scaled, lookback)
    if len(X) < 16:  # still too short after lookback
        raise RuntimeError(f"Insufficient data after lookback ({len(X)} samples) for LSTM training")

    # Train/val split
    split = int(0.8 * len(X))
    X_train, T_train = X[:split], T[:split]
    X_val, T_val = X[split:], T[split:] if split < len(X) else (X[:0], T[:0])

    # Tiny model
    model = keras.Sequential([
        layers.Input(shape=(lookback, 1)),
        layers.LSTM(32),
        layers.Dense(1)
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.005), loss="mse")

    callbacks = []
    if len(X_val) > 0:
        callbacks.append(keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        ))

    model.fit(
        X_train, T_train,
        validation_data=(X_val, T_val) if len(X_val) > 0 else None,
        epochs=60,
        batch_size=32,
        verbose=0,
        callbacks=callbacks
    )

    # Roll-ahead multi-step forecast
    def _forecast(h: int) -> np.ndarray:
        window = y_scaled[-lookback:].astype(float).copy()
        preds_scaled = []
        for _ in range(h):
            x = window.reshape(1, lookback, 1)
            yhat = float(model.predict(x, verbose=0)[0, 0])
            preds_scaled.append(yhat)
            # slide window
            window = np.concatenate([window[1:], np.array([yhat], dtype=float)], axis=0)
        preds = _unscale_minmax(np.array(preds_scaled, dtype=float), amin, amax)
        return preds

    return _forecast

# ---- Public API ------------------------------------------------------------

def build_daily_history_for_estimate(
    history_df: pd.DataFrame,   # columns: date, cloud, region, sku, unit_price
    estimate_bom,               # List[LineItem]
    cloud: str,
    region: str,
    days: int = 120,
) -> pd.Series:
    """
    Build the *historical* daily total cost (sum across SKUs) for the last `days`
    using real unit_price history x BoM daily quantity.
    """
    h = history_df[(history_df["cloud"] == cloud) & (history_df["region"] == region)].copy()
    if h.empty:
        return pd.Series(dtype=float)

    # wide table (date x sku) with continuous daily index
    pivot = _ensure_daily(h)

    # restrict to last `days`
    if len(pivot.index) > days:
        pivot = pivot.iloc[-days:]

    # restrict to last `days`
    last = pd.to_datetime(h["date"].max())
    cutoff = last - pd.Timedelta(days=days - 1)
    h = h[h["date"] >= cutoff]

    # pivot so each SKU has its unit_price column
    pivot = h.pivot_table(index="date", columns="sku", values="unit_price", aggfunc="last").sort_index().ffill()

    # build total daily cost using per-day qty for each line item
    total = pd.Series(0.0, index=pivot.index)
    for li in estimate_bom:
        sku = li.sku
        if sku not in pivot.columns:
            # if we don't have history for a SKU, skip it (or substitute current price later)
            continue
        qty_day = _per_day_qty(li.unit, li.qty)
        total += pivot[sku].astype(float) * float(qty_day)

    total.name = "daily_cost"
    return total

def forecast_total_cost(
    history_df: pd.DataFrame,
    estimate_bom,
    cloud: str,
    region: str,
    horizon_days: int = 90,
    seasonal: bool = False,
    model_name: str = "auto"  # <- NEW
) -> pd.Series:
    """
    Fit a forecaster per SKU unit_price, forecast unit prices h days ahead,
    convert to daily cost with BoM per-day quantities, and sum across SKUs.
    Returns a pd.Series indexed by future dates.
    """
    if history_df is None or history_df.empty:
        return pd.Series(dtype=float)

    # filter then make a continuous daily pivot (rows=date index, cols=sku)
    h = history_df[(history_df["cloud"] == cloud) & (history_df["region"] == region)].copy()
    if h.empty:
        return pd.Series(dtype=float)
    pivot = _ensure_daily(h)  # wide frame, index is DatetimeIndex named 'date'

    # fit per-SKU models
    sku_models: dict[str, callable] = {}
    for li in estimate_bom:
        sku = li.sku
        if sku not in pivot.columns:
            continue
        s = pivot[sku].dropna()
        if s.empty:
            continue
        forecaster = _fit_forecaster(s, seasonal=seasonal, model_name=model_name)  # <- pass through
        sku_models[sku] = forecaster

    if not sku_models:
        return pd.Series(dtype=float)

    # build future index from the last date in the pivot (index, not a column)
    last_date = pivot.index.max()
    future_idx = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon_days, freq="D")

    # sum forecasted daily costs across SKUs
    total_forecast = pd.Series(0.0, index=future_idx, name="forecast_cost")
    for li in estimate_bom:
        sku = li.sku
        if sku not in sku_models:
            continue
        qty_day = _per_day_qty(li.unit, li.qty)
        up = sku_models[sku](horizon_days)  # unit price prediction per day
        total_forecast += pd.Series(up, index=future_idx, dtype=float) * float(qty_day)

    return total_forecast