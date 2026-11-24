"""Tasks 12 and 14: turn forecasts into a daily replenishment decision and
quantify the money saved and the risk that remains.

Decision: every evening the store sets an order-up-to level S for tomorrow's
opening stock of Dettol 500 ml (overnight delivery from the distributor).
Unsold units carry over; demand above stock is lost.

Policies compared (buffer parameters tuned on VALIDATION, scored on TEST):
  A  Current practice: S = 7-day average x (1 + b)
  B  SARIMAX point forecast + k x validation RMSE
  C  LSTM point forecast + k x validation RMSE
  D  Quantile LSTM: S = forecast 97.5% quantile (+ m units, m tuned on val)
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import load_series, split, FIG, RES

# ---- cost assumptions (INR) - edit these to your store's numbers ----
PRICE = 220.0          # shelf price per bottle
MARGIN = 0.18          # retailer gross margin
GOODWILL = 15.0        # extra cost per lost sale (customer walks / basket loss)
C_UNDER = PRICE * MARGIN + GOODWILL          # ~ 54.6 per unit short
COST_PRICE = PRICE * (1 - MARGIN)
C_HOLD = COST_PRICE * 0.18 / 365 + 0.40      # capital + shelf space/handling per unit-day ~ 0.50
STORES = 150           # stores in the (hypothetical) chain for scale-up

s = load_series()
train, val, test = split(s)
cv = pd.read_csv(RES / "preds_val_classical.csv", index_col=0, parse_dates=True)
ct = pd.read_csv(RES / "preds_test_classical.csv", index_col=0, parse_dates=True)
nv = pd.read_csv(RES / "preds_val_neural.csv", index_col=0, parse_dates=True)
nt = pd.read_csv(RES / "preds_test_neural.csv", index_col=0, parse_dates=True)
ma7 = s.rolling(7).mean().shift(1)


def simulate(demand, S, start_stock=None):
    demand = np.asarray(demand, float); S = np.ceil(np.asarray(S, float))
    stock = S[0] if start_stock is None else start_stock
    lost = held = sold = 0.0; stockout_days = 0; ordered = 0.0
    lost_series = []
    for d, target in zip(demand, S):
        if stock < target:
            ordered += target - stock
            stock = target
        sale = min(d, stock)
        short = d - sale
        sold += sale; lost += short; stockout_days += short > 0
        stock -= sale
        held += stock            # units left overnight
        lost_series.append(short)
    cost_short, cost_hold = lost * C_UNDER, held * C_HOLD
    return {"fill_rate_%": sold / demand.sum() * 100, "stockout_days": int(stockout_days),
            "lost_units": lost, "avg_closing_stock": held / len(demand),
            "avg_opening_stock": float(np.mean(S)),
            "cost_lost_sales": cost_short, "cost_holding": cost_hold,
            "total_cost": cost_short + cost_hold, "lost_series": lost_series}


def levels(policy, param, which):
    if which == "val":
        y, c, n = val, cv, nv
    else:
        y, c, n = test, ct, nt
    if policy == "A":
        return ma7[y.index] * (1 + param)
    if policy == "B":
        sd = np.sqrt(np.mean((cv["actual"] - cv["SARIMAX + calendar"]) ** 2))
        return c["SARIMAX + calendar"] + param * sd
    if policy == "C":
        sd = np.sqrt(np.mean((nv["actual"] - nv["LSTM (tuned)"]) ** 2))
        return n["LSTM (tuned)"] + param * sd
    if policy == "D":
        return n["q975"] + param


grids = {"A": np.round(np.arange(0.0, 1.01, 0.05), 2), "B": np.round(np.arange(0, 4.51, 0.1), 2),
         "C": np.round(np.arange(0, 4.51, 0.1), 2), "D": np.arange(-10, 21, 1)}
names = {"A": "A. Current rule: 7-day avg x (1+b)", "B": "B. SARIMAX + k x RMSE buffer",
         "C": "C. LSTM + k x RMSE buffer", "D": "D. Quantile LSTM (97.5% upper bound)"}
out = {"assumptions": {"price": PRICE, "margin": MARGIN, "goodwill": GOODWILL,
                       "cost_under_per_unit": round(C_UNDER, 2), "cost_hold_per_unit_day": round(C_HOLD, 3),
                       "stores": STORES, "test_days": len(test)}, "policies": {}, "frontier": {}}

fig, ax = plt.subplots(figsize=(9, 5))
colors = {"A": "#7a7a7a", "B": "#2e86ab", "C": "#f18f01", "D": "#d1495b"}
for p, grid in grids.items():
    vals = [(g, simulate(val, levels(p, g, "val"))) for g in grid]
    best_g = min(vals, key=lambda t: t[1]["total_cost"])[0]
    # frontier on test for the chart
    tests_ = [simulate(test, levels(p, g, "test")) for g in grid]
    ax.plot([t["avg_closing_stock"] for t in tests_], [t["fill_rate_%"] for t in tests_],
            "-o", ms=3, color=colors[p], label=names[p])
    r = simulate(test, levels(p, best_g, "test"))
    ax.plot(r["avg_closing_stock"], r["fill_rate_%"], "*", ms=16, color=colors[p], mec="black")
    r.pop("lost_series")
    out["policies"][p] = dict(name=names[p], param=float(best_g), **{k: round(v, 2) for k, v in r.items()})
    # unbuffered version (pure point forecast) to show why intervals matter
    if p in ("B", "C"):
        r0 = simulate(test, levels(p, 0.0, "test")); r0.pop("lost_series")
        out["policies"][p + "0"] = dict(name=names[p].split(" +")[0] + " (no buffer)", param=0.0,
                                        **{k: round(v, 2) for k, v in r0.items()})
ax.set_xlabel("Average units left on shelf overnight (inventory carried)")
ax.set_ylabel("Fill rate on test period (% of demand served)")
ax.set_ylim(80, 100.5); ax.grid(alpha=0.3)
ax.set_title("Service vs inventory trade-off on the test period\n(stars = buffer chosen on validation set)")
ax.legend(frameon=False, loc="lower right", fontsize=8)
fig.tight_layout(); fig.savefig(FIG / "15_tradeoff.png", dpi=150); plt.close(fig)

# the rule of thumb as typically practised in stores (25% over last week's average)
rA = simulate(test, levels("A", 0.25, "test")); rA.pop("lost_series")
out["policies"]["A25"] = dict(name="A'. Rule of thumb as practised: 7-day avg x 1.25", param=0.25,
                              **{k: round(v, 2) for k, v in rA.items()})
base = out["policies"]["A"]
for p in ["B", "C", "D"]:
    r = out["policies"][p]
    save = base["total_cost"] - r["total_cost"]
    r["saving_vs_A_test_period"] = round(save, 0)
    r["saving_vs_A_per_store_year"] = round(save / len(test) * 365, 0)
    r["saving_vs_A_chain_year"] = round(save / len(test) * 365 * STORES, 0)
    r["saving_vs_A25_per_store_year"] = round((out["policies"]["A25"]["total_cost"] - r["total_cost"]) / len(test) * 365, 0)
    r["inventory_change_vs_A_%"] = round((r["avg_closing_stock"] / base["avg_closing_stock"] - 1) * 100, 1)

# Sensitivity of the annual per-store saving (B vs A) to the holding cost
sens = {}
for h in [0.25, 0.5, 1.0, 2.0]:
    C_HOLD_SAVE = C_HOLD
    globals()["C_HOLD"] = h
    best = {}
    for p in ["A", "B", "D"]:
        g = min(grids[p], key=lambda g: simulate(val, levels(p, g, "val"))["total_cost"])
        best[p] = simulate(test, levels(p, g, "test"))["total_cost"]
    sens[f"hold_{h}"] = {"B_vs_A": round((best["A"] - best["B"]) / len(test) * 365, 0),
                         "D_vs_A": round((best["A"] - best["D"]) / len(test) * 365, 0)}
    globals()["C_HOLD"] = C_HOLD_SAVE
out["sensitivity_holding_cost"] = sens

# ---------------- Task 14: risk ----------------
D = out["policies"]["D"]
S_D = levels("D", D["param"], "test")
sim = simulate(test, S_D)
lost = pd.Series(sim["lost_series"], index=test.index)
breach = (test > nt["q975"])
risk = {
    "days_demand_above_upper": int(breach.sum()),
    "breach_dates": [str(d.date()) for d in test.index[breach]],
    "breach_in_aug": int(breach[test.index.month == 8].sum()),
    "breach_in_pre_diwali_2025": int(breach["2025-10-08":"2025-10-20"].sum()),
    "max_daily_shortfall_units": float(lost.max()),
    "expected_daily_shortfall_cost": round(lost.mean() * C_UNDER, 2),
    "worst_day_cost": round(lost.max() * C_UNDER, 2),
    "lower_bound_use": "q2.5 used as a floor: if opening stock falls below q2.5, trigger emergency top-up",
}
# Exposure: cost if forecast is biased by -10% / +10% (model drift)
for bias in [-0.20, -0.10, 0.10]:
    r = simulate(test, S_D * (1 + bias)); r.pop("lost_series")
    risk[f"bias_{int(bias*100)}%"] = {k: round(r[k], 2) for k in ["fill_rate_%", "stockout_days", "total_cost"]}
# Bootstrap 90% range of annual per-store cost for policy D vs A (block bootstrap by week)
rng = np.random.default_rng(0)
weeks = np.arange(len(test)) // 7
SA = levels("A", out["policies"]["A"]["param"], "test").values
SDv = S_D.values; y = test.values
diffs = []
for _ in range(1000):
    pick = rng.choice(np.unique(weeks), size=len(np.unique(weeks)), replace=True)
    ix = np.concatenate([np.where(weeks == w)[0] for w in pick])
    a = simulate(y[ix], SA[ix])["total_cost"]; d = simulate(y[ix], SDv[ix])["total_cost"]
    diffs.append((a - d) / len(ix) * 365)
risk["annual_saving_per_store_P5_P50_P95"] = [round(np.percentile(diffs, q), 0) for q in (5, 50, 95)]
out["risk"] = risk

# risk chart: interval, stock level, shortfalls
fig, ax = plt.subplots(figsize=(11, 4))
ax.fill_between(test.index, nt["q025"], nt["q975"], color="#2e86ab", alpha=0.2, label="95% interval (quantile LSTM)")
ax.plot(test.index, test, color="black", lw=1, label="Actual demand")
ax.step(test.index, np.ceil(S_D), where="mid", color="#d1495b", lw=1, label="Opening stock (policy D)")
ax.scatter(test.index[breach], test[breach], color="#d1495b", zorder=5, s=25, label="Demand above upper bound")
ax.set_ylabel("Units"); ax.grid(alpha=0.3); ax.legend(frameon=False, ncol=4, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.08))
ax.set_title("Policy D on the test period: where the interval was breached")
fig.tight_layout(); fig.savefig(FIG / "16_risk.png", dpi=150); plt.close(fig)

json.dump(out, open(RES / "decision.json", "w"), indent=2, default=float)
print(json.dumps(out, indent=1, default=float))
