"""Tasks 4-9: MA(q), AR(p), unit-root tests, differencing, ARIMA, SARIMA(X).

Protocol for every model: orders are chosen on TRAIN (+ AIC) and VALIDATION
(one-step RMSE); parameters are estimated on TRAIN only; the fitted model is
then run forward over val+test WITHOUT re-estimation, so each test prediction
uses only data up to the previous day (one-step-ahead)."""
import json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from common import load_series, split, metrics, plot_pred, FIG, RES

warnings.filterwarnings("ignore")
s = load_series()
train, val, test = split(s)
out = {"selection": {}, "metrics": {}, "params": {}}
preds = pd.DataFrame({"actual": test}, index=test.index)
val_preds = pd.DataFrame({"actual": val}, index=val.index)


def one_step(res, exog_full=None):
    """Extend fitted results over the whole series with fixed parameters and
    return one-step-ahead predictions for val and test."""
    full = res.apply(s, exog=exog_full) if exog_full is not None else res.apply(s)
    p = full.predict(start=val.index[0], end=test.index[-1])
    return p[val.index], p[test.index]


def fit(order, seasonal=(0, 0, 0, 0), trend="c", exog=None):
    m = SARIMAX(train, order=order, seasonal_order=seasonal, trend=trend,
                exog=None if exog is None else exog.loc[train.index])
    return m.fit(disp=False, maxiter=200)


# ---------------- Task 4: MA(q) ----------------
rows = []
for q in [1, 2, 3, 5, 7, 14]:
    r = fit((0, 0, q))
    pv, _ = one_step(r)
    rows.append({"q": q, "AIC": round(r.aic, 1), "val_RMSE": round(metrics(val, pv)["RMSE"], 3)})
out["selection"]["MA"] = rows
q_best = min(rows, key=lambda d: d["val_RMSE"])["q"]
r = fit((0, 0, q_best))
pv, pt = one_step(r)
name = f"MA({q_best})"
preds[name], val_preds[name] = pt, pv
out["metrics"][name] = metrics(test, pt)
plot_pred(test.index, test, pt, f"{name}: one-step-ahead forecast on test set", "05_ma.png")

# ---------------- Task 5: AR(p) ----------------
rows = []
for p in [1, 2, 7, 8, 14, 21, 28]:
    r = fit((p, 0, 0))
    pv, _ = one_step(r)
    rows.append({"p": p, "AIC": round(r.aic, 1), "val_RMSE": round(metrics(val, pv)["RMSE"], 3)})
out["selection"]["AR"] = rows
p_best = min(rows, key=lambda d: d["val_RMSE"])["p"]
r = fit((p_best, 0, 0))
pv, pt = one_step(r)
name = f"AR({p_best})"
preds[name], val_preds[name] = pt, pv
out["metrics"][name] = metrics(test, pt)
plot_pred(test.index, test, pt, f"{name}: one-step-ahead forecast on test set", "06_ar.png")

# ---------------- Task 6/7: stationarity ----------------
def tests(x, label):
    x = x.dropna()
    adf_c = adfuller(x, regression="c", autolag="AIC")
    adf_ct = adfuller(x, regression="ct", autolag="AIC")
    kp_c = kpss(x, regression="c", nlags="auto")
    kp_ct = kpss(x, regression="ct", nlags="auto")
    return {"series": label,
            "ADF_c_stat": round(adf_c[0], 3), "ADF_c_p": round(adf_c[1], 4), "ADF_c_lags": adf_c[2],
            "ADF_ct_stat": round(adf_ct[0], 3), "ADF_ct_p": round(adf_ct[1], 4),
            "KPSS_c_stat": round(kp_c[0], 3), "KPSS_c_p": round(kp_c[1], 3),
            "KPSS_ct_stat": round(kp_ct[0], 3), "KPSS_ct_p": round(kp_ct[1], 3)}

d1 = train.diff()
d7 = train.diff(7)
d1d7 = train.diff(7).diff()
out["stationarity"] = [tests(train, "Level (train)"), tests(d1, "First difference"),
                       tests(d7, "Seasonal difference (lag 7)"),
                       tests(d1d7, "First + seasonal difference")]
# rolling statistics plot
fig, axes = plt.subplots(3, 1, figsize=(11, 7.5), sharex=True)
for ax, (x, lab) in zip(axes, [(train, "Level"), (d7, "Seasonal difference (lag 7)"),
                              (d1d7, "First + seasonal difference")]):
    ax.plot(x, lw=0.7, color="#888888", label=lab)
    ax.plot(x.rolling(56).mean(), color="#d1495b", lw=1.5, label="56-day rolling mean")
    ax.plot(x.rolling(56).std(), color="#2e86ab", lw=1.5, label="56-day rolling std")
    ax.legend(frameon=False, ncol=3, loc="upper left", fontsize=8); ax.grid(alpha=0.3)
axes[0].set_title("Training series before and after differencing")
fig.tight_layout(); fig.savefig(FIG / "07_stationarity.png", dpi=150); plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
plot_acf(d7.dropna(), lags=40, ax=axes[0], title="ACF after seasonal difference (lag 7)")
plot_pacf(d7.dropna(), lags=40, ax=axes[1], method="ywm", title="PACF after seasonal difference (lag 7)")
for a in axes: a.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(FIG / "08_acf_after_diff.png", dpi=150); plt.close(fig)

# ---------------- Task 8: ARIMA(p,1,q) ----------------
rows = []
for p in [0, 1, 2, 7]:
    for q in [0, 1, 2, 7]:
        if p == 0 and q == 0:
            continue
        try:
            r = fit((p, 1, q), trend="n")
            pv, _ = one_step(r)
            rows.append({"p": p, "d": 1, "q": q, "AIC": round(r.aic, 1),
                         "val_RMSE": round(metrics(val, pv)["RMSE"], 3)})
        except Exception as e:
            print("skip", p, q, e)
out["selection"]["ARIMA"] = rows
best = min(rows, key=lambda d: d["val_RMSE"])
r = fit((best["p"], 1, best["q"]), trend="n")
pv, pt = one_step(r)
name = f"ARIMA({best['p']},1,{best['q']})"
preds[name], val_preds[name] = pt, pv
out["metrics"][name] = metrics(test, pt)
out["params"][name] = {k: round(v, 4) for k, v in r.params.items()}
plot_pred(test.index, test, pt, f"{name}: one-step-ahead forecast on test set", "09_arima.png")

# ---------------- Task 9: SARIMA(p,0,q)(P,1,Q)_7 ----------------
rows = []
for (p, q) in [(1, 0), (1, 1), (2, 1)]:
    for (P, Q) in [(0, 1), (1, 1)]:
        r = fit((p, 0, q), (P, 1, Q, 7), trend="n")
        pv, _ = one_step(r)
        rows.append({"order": f"({p},0,{q})({P},1,{Q})7", "AIC": round(r.aic, 1),
                     "val_RMSE": round(metrics(val, pv)["RMSE"], 3), "pq": (p, q), "PQ": (P, Q)})
out["selection"]["SARIMA"] = [{k: v for k, v in d.items() if k not in ("pq", "PQ")} for d in rows]
best = min(rows, key=lambda d: d["val_RMSE"])
r_sar = fit((best["pq"][0], 0, best["pq"][1]), (best["PQ"][0], 1, best["PQ"][1], 7), trend="n")
pv, pt = one_step(r_sar)
name = f"SARIMA{best['order']}"
preds[name], val_preds[name] = pt, pv
out["metrics"][name] = metrics(test, pt)
out["params"][name] = {k: round(v, 4) for k, v in r_sar.params.items()}
plot_pred(test.index, test, pt, f"{name}: one-step-ahead forecast on test set", "10_sarima.png")
resid = r_sar.resid[14:]
from statsmodels.stats.diagnostic import acorr_ljungbox
out["sarima_resid_ljungbox_p"] = {int(k): float(v) for k, v in
                                  acorr_ljungbox(resid, lags=[7, 14, 28])["lb_pvalue"].items()}

# ------ Extension: SARIMAX with calendar regressors known in advance ------
idx = s.index
doy = idx.dayofyear.values
diwali = pd.to_datetime(["2023-11-12", "2024-11-01", "2025-10-20"])
pre_diwali = np.zeros(len(idx))
for d in diwali:
    db = (d - idx).days.values
    m = (db >= 0) & (db <= 12)
    pre_diwali[m] = 1 - db[m] / 13
exog = pd.DataFrame({
    "sin1": np.sin(2 * np.pi * doy / 365.25), "cos1": np.cos(2 * np.pi * doy / 365.25),
    "sin2": np.sin(4 * np.pi * doy / 365.25), "cos2": np.cos(4 * np.pi * doy / 365.25),
    "month_start": (idx.day <= 5).astype(float), "pre_diwali": pre_diwali}, index=idx)
exog_d = exog.diff(7).fillna(0)  # regressors enter after the seasonal difference
r_x = SARIMAX(train, exog=exog.loc[train.index], order=(best["pq"][0], 0, best["pq"][1]),
              seasonal_order=(best["PQ"][0], 1, best["PQ"][1], 7), trend="n").fit(disp=False, maxiter=300)
pv, pt = one_step(r_x, exog_full=exog)
name = "SARIMAX + calendar"
preds[name], val_preds[name] = pt, pv
out["metrics"][name] = metrics(test, pt)
out["params"][name] = {k: round(v, 4) for k, v in r_x.params.items()}
plot_pred(test.index, test, pt, "SARIMAX with calendar regressors: one-step-ahead forecast on test set",
          "11_sarimax.png")
exog.to_csv(RES / "calendar_exog.csv")

# naive benchmarks for context
nv = s.shift(1); sn = s.shift(7)
preds["Naive (yesterday)"], val_preds["Naive (yesterday)"] = nv[test.index], nv[val.index]
preds["Seasonal naive (last week)"], val_preds["Seasonal naive (last week)"] = sn[test.index], sn[val.index]
out["metrics"]["Naive (yesterday)"] = metrics(test, nv[test.index])
out["metrics"]["Seasonal naive (last week)"] = metrics(test, sn[test.index])

preds.to_csv(RES / "preds_test_classical.csv")
val_preds.to_csv(RES / "preds_val_classical.csv")
json.dump(out, open(RES / "classical.json", "w"), indent=2, default=str)
print(json.dumps(out["selection"], indent=0, default=str))
print(json.dumps(out["stationarity"], indent=0))
for k, v in out["metrics"].items():
    print(f"{k:32s}", {a: round(b, 3) for a, b in v.items()})
print(out["params"])
print(out["sarima_resid_ljungbox_p"])
