"""Collect all test-set metrics into one table and one comparison chart."""
import json
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import RES, FIG

c = json.load(open(RES / "classical.json"))["metrics"]
n = json.load(open(RES / "neural.json"))["metrics"]
rows = {**c, **{k: v for k, v in n.items() if not k.endswith(" val")}}
df = pd.DataFrame(rows).T.round(3).sort_values("RMSE")
df.to_csv(RES / "metrics_test_all_models.csv")
print(df)
fig, ax = plt.subplots(figsize=(9, 4.2))
cols = ["#d1495b" if "SARIMAX" in i else "#2e86ab" if any(x in i for x in ["MA", "AR", "SARIMA"]) and "Naive" not in i
        else "#f18f01" if any(x in i for x in ["RNN", "LSTM"]) else "#999999" for i in df.index]
ax.barh(df.index[::-1], df["RMSE"][::-1], color=cols[::-1])
for i, v in enumerate(df["RMSE"][::-1]):
    ax.text(v + 0.1, i, f"{v:.2f}", va="center", fontsize=8)
ax.set_xlabel("Test RMSE (units/day), lower is better"); ax.grid(axis="x", alpha=0.3)
ax.set_title("One-step-ahead accuracy on the test period (Jul-Dec 2025)")
fig.tight_layout(); fig.savefig(FIG / "17_model_comparison.png", dpi=150); plt.close(fig)
