# Forecasting daily sales of a Dettol SKU to set store replenishment levels

Course project: time-series forecasting (tasks 1–14).
SKU: **Dettol Antiseptic Liquid 500 ml**, one supermarket (store `PUNE-017`), daily unit sales 2023-01-01 to 2025-12-31.

> **Data note.** Store-level POS data for a single SKU is not publicly available, so `data/dettol_500ml_daily_sales.csv`
> is **simulated** by `src/00_generate_data.py` from realistic demand drivers (trend, weekday pattern, salary-day effect,
> monsoon peak, Diwali cleaning surge, persistent shocks and Poisson noise). Drop in a real CSV with the same columns
> (`date, sku, store_id, units_sold`) and every script runs unchanged.

## Decision supported
Every evening the store sets tomorrow's **order-up-to level** for the SKU. The forecast (plus a buffer from its error or
from a quantile model) replaces a rule of thumb (last 7-day average × 1.25).

## Repository layout
```
data/      dettol_500ml_daily_sales.csv
src/       common.py (split, metrics, plotting)
           00_generate_data.py   simulate the series
           01_eda.py             decomposition, ACF/PACF, split            (tasks 1-3)
           02_classical.py       MA, AR, ADF/KPSS, ARIMA, SARIMA, SARIMAX  (tasks 4-9)
           03_neural.py          RNN, LSTM tuning, quantile LSTMs          (tasks 10, 11, 13)
           04_decision.py        inventory simulation, savings and risk    (tasks 12, 14)
           05_summary.py         metrics table and comparison chart
figures/   all plots used in the report
results/   JSON/CSV outputs (metrics, tuning tables, predictions)
```

## Split (same for every model, no shuffling)
| Set | Dates | Days |
|---|---|---|
| Train | 2023-01-01 – 2024-12-31 | 731 |
| Validation | 2025-01-01 – 2025-06-30 | 181 |
| Test | 2025-07-01 – 2025-12-31 | 184 |

All models make one-step-ahead (next-day) predictions; parameters are estimated on train only.

## Reproduce
```bash
pip install -r requirements.txt
./run_all.sh
```

## Headline results (test period)
| Model | RMSE | MAE | MAPE % | R² |
|---|---|---|---|---|
| SARIMAX + calendar | 9.17 | 7.38 | 14.3 | 0.35 |
| ARIMA(7,1,7) | 9.45 | 7.65 | 15.4 | 0.31 |
| SARIMA(1,0,0)(0,1,1)7 | 9.88 | 7.98 | 15.6 | 0.25 |
| LSTM (tuned) | 10.04 | 8.09 | 15.4 | 0.22 |
| AR(28) | 10.09 | 8.09 | 15.6 | 0.22 |
| RNN (tuned) | 10.54 | 8.49 | 15.8 | 0.14 |
| MA(14) | 11.51 | 8.99 | 16.6 | −0.02 |
| Naive (yesterday) | 13.36 | 10.52 | 21.3 | −0.38 |

Quantile LSTM 95% interval: PICP 92.9 %, MPIW 40.4 units.
Replacing the rule of thumb with SARIMAX + 3×RMSE buffer: fill rate 97.9 % → 99.9 %, about ₹19,000 per store per year.
