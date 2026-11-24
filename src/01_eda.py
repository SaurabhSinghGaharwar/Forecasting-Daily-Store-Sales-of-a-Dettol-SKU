"""Tasks 1-3: describe the data, decompose it, autocorrelations, split."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.seasonal import MSTL, seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from common import load_series, split, FIG, RES

s = load_series()
train, val, test = split(s)
out = {}

# Raw series with split
fig, ax = plt.subplots(figsize=(11, 3.8))
for part, c, lab in [(train, "#2e86ab", "Train"), (val, "#f18f01", "Validation"), (test, "#d1495b", "Test")]:
    ax.plot(part.index, part.values, color=c, lw=0.9, label=f"{lab} ({len(part)} days)")
ax.plot(s.rolling(30, center=True).mean(), color="black", lw=1.6, label="30-day mean")
ax.set_title("Daily sales of Dettol Antiseptic Liquid 500 ml, store PUNE-017")
ax.set_ylabel("Units sold per day"); ax.grid(alpha=0.3); ax.legend(frameon=False, ncol=4, loc="upper left")
fig.tight_layout(); fig.savefig(FIG / "01_series_split.png", dpi=150); plt.close(fig)

# Descriptives
out["n_days"] = len(s)
out["split"] = {k: [str(p.index[0].date()), str(p.index[-1].date()), len(p)] for k, p in
                [("train", train), ("val", val), ("test", test)]}
out["describe"] = s.describe().round(2).to_dict()
yearly = s.groupby(s.index.year).mean().round(2)
out["yearly_mean"] = {int(k): v for k, v in yearly.items()}
dow = s.groupby(s.index.dayofweek).mean()
out["dow_mean"] = dict(zip(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], dow.round(2)))
mon = s.groupby(s.index.month).mean()
out["month_mean"] = {int(k): round(v, 2) for k, v in mon.items()}
out["month_start_vs_rest"] = [round(s[s.index.day <= 5].mean(), 2), round(s[s.index.day > 5].mean(), 2)]

# Classical additive decomposition (weekly) and MSTL (weekly + yearly)
dec = seasonal_decompose(s, model="additive", period=7)
fig = dec.plot(); fig.set_size_inches(11, 7)
fig.suptitle("Classical additive decomposition (period = 7 days)", y=1.0)
fig.tight_layout(); fig.savefig(FIG / "02_decompose_weekly.png", dpi=150); plt.close(fig)

mstl = MSTL(s, periods=(7, 365)).fit()
# With only three yearly cycles MSTL's day-level yearly component soaks up noise,
# so we smooth it (31-day centred mean) and put the remainder back into the residual.
weekly = mstl.seasonal["seasonal_7"]
yearly = mstl.seasonal["seasonal_365"].rolling(31, center=True, min_periods=1).mean()
trend = mstl.trend
resid_s = s - trend - weekly - yearly
fig, axes = plt.subplots(5, 1, figsize=(11, 10), sharex=True)
for ax, (name, comp) in zip(axes, [("Observed", s), ("Trend", trend),
                                   ("Weekly seasonality", weekly),
                                   ("Yearly seasonality\n(31-day smoothed)", yearly),
                                   ("Residual (noise)", resid_s)]):
    ax.plot(comp, lw=0.8, color="#2e86ab" if name != "Residual (noise)" else "#666666")
    ax.set_ylabel(name, fontsize=8); ax.grid(alpha=0.3)
axes[0].set_title("MSTL decomposition: trend + weekly + yearly seasonality + residual")
fig.tight_layout(); fig.savefig(FIG / "03_decompose_mstl.png", dpi=150); plt.close(fig)

resid = resid_s.dropna()
var_s = s.var()
out["mstl_variance_share_%"] = {
    "trend": round(trend.var() / var_s * 100, 1),
    "weekly": round(weekly.var() / var_s * 100, 1),
    "yearly": round(yearly.var() / var_s * 100, 1),
    "residual": round(resid.var() / var_s * 100, 1)}
out["trend_start_end"] = [round(trend.iloc[0], 2), round(trend.iloc[-1], 2)]
out["yearly_peak_trough"] = [str(yearly.idxmax().date()), round(yearly.max(), 2), str(yearly.idxmin().date()), round(yearly.min(), 2)]
out["weekly_profile"] = dict(zip(["Mon","Tue","Wed","Thu","Fri","Sat","Sun"], weekly.groupby(weekly.index.dayofweek).mean().round(2)))
out["resid_mean_sd"] = [round(resid.mean(), 3), round(resid.std(), 3)]
lb = acorr_ljungbox(resid, lags=[7, 14, 28])
out["resid_ljungbox_p"] = {int(k): float(v) for k, v in lb["lb_pvalue"].items()}
out["resid_acf_1_7"] = [round(x, 3) for x in acf(resid, nlags=7)[1:]]

# ACF / PACF of raw series and residual
fig, axes = plt.subplots(2, 2, figsize=(11, 6.5))
plot_acf(s, lags=60, ax=axes[0, 0], title="ACF: raw sales (lags 0-60)")
plot_pacf(s, lags=60, ax=axes[0, 1], method="ywm", title="PACF: raw sales")
plot_acf(resid, lags=60, ax=axes[1, 0], title="ACF: MSTL residual")
plot_pacf(resid, lags=60, ax=axes[1, 1], method="ywm", title="PACF: MSTL residual")
for a in axes.flat: a.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(FIG / "04_acf_pacf.png", dpi=150); plt.close(fig)

a = acf(s, nlags=365)
out["acf_raw"] = {int(k): round(a[k], 3) for k in [1, 2, 3, 6, 7, 14, 21, 28, 30, 182, 364, 365]}
p = pacf(s, nlags=30, method="ywm")
out["pacf_raw"] = {int(k): round(p[k], 3) for k in range(1, 15)}
out["pacf_raw_sig_lags_1_30"] = [int(k) for k in range(1, 31) if abs(p[k]) > 1.96 / np.sqrt(len(s))]

json.dump(out, open(RES / "eda.json", "w"), indent=2)
print(json.dumps(out, indent=1))
