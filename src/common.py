"""Shared paths, data split and metrics used by every script."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "dettol_500ml_daily_sales.csv"
FIG = ROOT / "figures"
RES = ROOT / "results"
FIG.mkdir(exist_ok=True)
RES.mkdir(exist_ok=True)

# Chronological split (no shuffling) used by EVERY model
TRAIN_END = "2024-12-31"
VAL_END = "2025-06-30"   # test = 2025-07-01 .. 2025-12-31


def load_series():
    df = pd.read_csv(DATA, parse_dates=["date"], index_col="date")
    s = df["units_sold"].astype(float)
    s.index.freq = "D"
    return s


def split(s):
    train = s[:TRAIN_END]
    val = s[pd.Timestamp(TRAIN_END) + pd.Timedelta(days=1):VAL_END]
    test = s[pd.Timestamp(VAL_END) + pd.Timedelta(days=1):]
    return train, val, test


def metrics(y, yhat):
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    err = y - yhat
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    mape = float(np.mean(np.abs(err / y)) * 100)
    r2 = float(1 - np.sum(err ** 2) / np.sum((y - y.mean()) ** 2))
    return {"RMSE": rmse, "MAE": mae, "MAPE_%": mape, "R2": r2}


def plot_pred(idx, y, yhat, title, fname, lower=None, upper=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.plot(idx, y, color="#222222", lw=1.2, label="Actual")
    ax.plot(idx, yhat, color="#d1495b", lw=1.2, label="Predicted (one-step-ahead)")
    if lower is not None:
        ax.fill_between(idx, lower, upper, color="#2e86ab", alpha=0.25,
                        label="95% prediction interval")
    ax.set_title(title)
    ax.set_ylabel("Units sold per day")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / fname, dpi=150)
    plt.close(fig)
