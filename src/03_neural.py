"""Tasks 10, 11 and 13: RNN, LSTM (tuned on validation) and a pair of
quantile LSTMs trained with the pinball loss (95% prediction interval).

Inputs are the last L days of sales (min-max scaled with TRAIN statistics);
output is next-day sales (one-step-ahead). Same train/val/test split as the
classical models."""
import json, random, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from common import load_series, split, metrics, plot_pred, RES

torch.set_num_threads(2)
s = load_series()
train, val, test = split(s)
lo, hi = train.min(), train.max()
z = ((s - lo) / (hi - lo)).values.astype(np.float32)
n_tr, n_va = len(train), len(val)
i_val0, i_test0 = n_tr, n_tr + n_va


def windows(L, start, end):
    """X[t] = z[t-L:t], y[t] = z[t] for target indices start..end-1."""
    idx = np.arange(max(start, L), end)
    X = np.stack([z[i - L:i] for i in idx])[..., None]
    return torch.tensor(X), torch.tensor(z[idx]), idx


class Net(nn.Module):
    def __init__(self, cell, hidden, layers):
        super().__init__()
        rnn = nn.RNN if cell == "RNN" else nn.LSTM
        self.rnn = rnn(1, hidden, num_layers=layers, batch_first=True,
                       dropout=0.1 if layers > 1 else 0.0)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        o, _ = self.rnn(x)
        return self.head(o[:, -1]).squeeze(-1)


def pinball(tau):
    def f(pred, y):
        e = y - pred
        return torch.mean(torch.maximum(tau * e, (tau - 1) * e))
    return f


def train_model(cfg, seed=0, loss_fn=None, max_epochs=200, patience=20):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    L = cfg["lag"]
    Xtr, ytr, _ = windows(L, 0, i_val0)
    Xva, yva, _ = windows(L, i_val0, i_test0)
    net = Net(cfg["cell"], cfg["hidden"], cfg["layers"])
    opt = torch.optim.Adam(net.parameters(), lr=cfg["lr"])
    loss_fn = loss_fn or nn.MSELoss()
    best, best_state, wait = np.inf, None, 0
    for ep in range(max_epochs):
        net.train()
        perm = torch.randperm(len(Xtr))
        for b in range(0, len(Xtr), cfg["batch"]):
            j = perm[b:b + cfg["batch"]]
            opt.zero_grad()
            l = loss_fn(net(Xtr[j]), ytr[j])
            l.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            vl = loss_fn(net(Xva), yva).item()
        if vl < best - 1e-6:
            best, wait = vl, 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    net.load_state_dict(best_state)
    return net, ep + 1


def predict(net, L, start, end):
    X, _, idx = windows(L, start, end)
    net.eval()
    with torch.no_grad():
        p = net(X).numpy()
    return p * (hi - lo) + lo, idx


space = {"lag": [7, 14, 28], "batch": [16, 32, 64], "lr": [1e-3, 3e-3, 1e-2],
         "layers": [1, 2], "hidden": [16, 32, 64]}
rng = random.Random(7)
configs = []
seen = set()
while len(configs) < 24:
    c = tuple(rng.choice(v) for v in space.values())
    if c not in seen:
        seen.add(c); configs.append(dict(zip(space.keys(), c)))

out = {"tuning": {}, "best": {}, "metrics": {}, "interval": {}}
preds = pd.DataFrame({"actual": test}, index=test.index)
val_preds = pd.DataFrame({"actual": val}, index=val.index)

for cell in ["RNN", "LSTM"]:
    rows = []
    t0 = time.time()
    for c in configs:
        cfg = dict(c, cell=cell)
        net, ep = train_model(cfg)
        pv, _ = predict(net, cfg["lag"], i_val0, i_test0)
        rows.append(dict(c, epochs=ep, val_RMSE=round(metrics(val, pv)["RMSE"], 3)))
    rows.sort(key=lambda d: d["val_RMSE"])
    out["tuning"][cell] = rows
    best = {k: rows[0][k] for k in space}
    out["best"][cell] = best
    print(cell, "tuned in", round(time.time() - t0), "s; best", rows[0])
    # final: average of 3 seeds with the best configuration
    cfg = dict(best, cell=cell)
    pv_all, pt_all = [], []
    for seed in range(3):
        net, _ = train_model(cfg, seed=seed)
        pv_all.append(predict(net, cfg["lag"], i_val0, i_test0)[0])
        pt_all.append(predict(net, cfg["lag"], i_test0, len(s))[0])
    pv, pt = np.mean(pv_all, 0), np.mean(pt_all, 0)
    name = f"{cell} (tuned)"
    preds[name], val_preds[name] = pt, pv
    out["metrics"][name] = metrics(test, pt)
    out["metrics"][name + " val"] = metrics(val, pv)
    plot_pred(test.index, test, pt, f"{name}: one-step-ahead forecast on test set "
              f"(lag={best['lag']}, layers={best['layers']}, units={best['hidden']}, "
              f"batch={best['batch']}, lr={best['lr']})",
              "12_rnn.png" if cell == "RNN" else "13_lstm.png")

# ---------------- Task 13: quantile LSTMs (pinball loss) ----------------
cfg = dict(out["best"]["LSTM"], cell="LSTM")
q = {}
for tau in [0.025, 0.975]:
    reps_v, reps_t = [], []
    for seed in range(3):
        net, _ = train_model(cfg, seed=seed, loss_fn=pinball(tau))
        reps_v.append(predict(net, cfg["lag"], i_val0, i_test0)[0])
        reps_t.append(predict(net, cfg["lag"], i_test0, len(s))[0])
    q[tau] = (np.mean(reps_v, 0), np.mean(reps_t, 0))
lo_v, hi_v = np.minimum(q[0.025][0], q[0.975][0]), np.maximum(q[0.025][0], q[0.975][0])
lo_t, hi_t = np.minimum(q[0.025][1], q[0.975][1]), np.maximum(q[0.025][1], q[0.975][1])


def picp_mpiw(y, l, u):
    y = np.asarray(y)
    return {"PICP_%": float(np.mean((y >= l) & (y <= u)) * 100), "MPIW_units": float(np.mean(u - l)),
            "below_lower_%": float(np.mean(y < l) * 100), "above_upper_%": float(np.mean(y > u) * 100)}


out["interval"]["val"] = picp_mpiw(val, lo_v, hi_v)
out["interval"]["test"] = picp_mpiw(test, lo_t, hi_t)
preds["q025"], preds["q975"] = lo_t, hi_t
val_preds["q025"], val_preds["q975"] = lo_v, hi_v
plot_pred(test.index, test, preds["LSTM (tuned)"], "Quantile LSTM 95% prediction interval "
          f"(PICP={out['interval']['test']['PICP_%']:.1f}%, MPIW={out['interval']['test']['MPIW_units']:.1f} units)",
          "14_quantile_interval.png", lower=lo_t, upper=hi_t)

preds.to_csv(RES / "preds_test_neural.csv")
val_preds.to_csv(RES / "preds_val_neural.csv")
json.dump(out, open(RES / "neural.json", "w"), indent=2)
for k, v in out["metrics"].items():
    print(f"{k:22s}", {a: round(b, 3) for a, b in v.items()})
print(out["interval"])
