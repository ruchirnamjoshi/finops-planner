from __future__ import annotations
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
from .schemas import Estimate
import pandas as pd

def savings_waterfall(base: Estimate, opt: Estimate):
    fig = plt.figure()
    plt.title("Savings Waterfall")
    plt.bar(["Base","Optimized"], [base.monthly_cost, opt.monthly_cost])
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    img = Image.open(buf).convert("RGBA")
    buf.close()
    plt.close(fig)
    return img


def _to_pil(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    img = Image.open(buf).convert("RGBA")
    buf.close()
    plt.close(fig)
    return img

def savings_waterfall(base: Estimate, opt: Estimate):
    fig = plt.figure()
    plt.title("Savings Waterfall")
    plt.bar(["Base","Optimized"], [base.monthly_cost, opt.monthly_cost])
    return _to_pil(fig)

def forecast_line(history: pd.Series, forecast: pd.Series):
    fig = plt.figure()
    plt.title("Daily Cost: history & forecast")
    if history is not None and not history.empty:
        history.sort_index(inplace=True)
        plt.plot(history.index, history.values, label="History")
    if forecast is not None and not forecast.empty:
        forecast.sort_index(inplace=True)
        plt.plot(forecast.index, forecast.values, linestyle="--", label="Forecast")
    plt.legend()
    return _to_pil(fig)

def cost_breakdown_bar(est: Estimate):
    # simple cost by service chart for the winner
    by_service = {}
    for li in est.bom:
        by_service[li.service] = by_service.get(li.service, 0.0) + li.cost
    if not by_service:
        return None
    s = pd.Series(by_service).sort_values(ascending=False)
    fig = plt.figure()
    plt.title("Cost Breakdown by Service")
    plt.bar(s.index, s.values)
    plt.xticks(rotation=30, ha="right")
    return _to_pil(fig)