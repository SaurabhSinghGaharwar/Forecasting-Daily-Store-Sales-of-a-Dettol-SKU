"""Simulate daily unit sales of Dettol Antiseptic Liquid 500 ml in one
mid-sized supermarket (2023-01-01 to 2025-12-31).

Real store-level POS data for a single SKU is not publicly available, so the
series is simulated from documented, realistic demand drivers:
  * slow growth in category demand (~8% a year)
  * day-of-week pattern (weekend shopping peak)
  * month-start salary effect (days 1-5)
  * monsoon peak (Jul-Sep: more infections, more disinfectant use)
  * mild winter cold/flu bump (Dec-Jan)
  * Diwali deep-cleaning surge in the ~12 days before Diwali
  * persistent demand shocks (AR(1) in log space, e.g. local outbreaks)
  * Poisson sampling noise (integer unit sales)
Replace data/dettol_500ml_daily_sales.csv with real POS data and every other
script runs unchanged.
"""
import numpy as np
import pandas as pd
from common import DATA

rng = np.random.default_rng(42)
dates = pd.date_range("2023-01-01", "2025-12-31", freq="D")
n = len(dates)
t = np.arange(n)

level = 38 * (1 + 0.08 * t / 365)
dow = np.array([0.90, 0.88, 0.92, 0.95, 1.02, 1.18, 1.15])[dates.dayofweek]
salary = np.where(dates.day <= 5, 1.08, 1.0)

doy = dates.dayofyear.values
monsoon = 1 + 0.18 * np.exp(-0.5 * ((doy - 222) / 28) ** 2)
winter = 1 + 0.06 * (np.exp(-0.5 * ((doy - 10) / 20) ** 2)
                     + np.exp(-0.5 * ((doy - 375) / 20) ** 2))

diwali = pd.to_datetime(["2023-11-12", "2024-11-01", "2025-10-20"])
festive = np.ones(n)
for d in diwali:
    days_before = (d - dates).days.values
    mask = (days_before >= 0) & (days_before <= 12)
    festive[mask] += 0.35 * (1 - days_before[mask] / 13)

shock = np.zeros(n)
eps = rng.normal(0, 0.07, n)
for i in range(1, n):
    shock[i] = 0.6 * shock[i - 1] + eps[i]

mean = level * dow * salary * monsoon * winter * festive * np.exp(shock)
units = rng.poisson(mean)

pd.DataFrame({"date": dates, "sku": "Dettol Antiseptic Liquid 500ml",
              "store_id": "PUNE-017", "units_sold": units}).to_csv(DATA, index=False)
print(f"Saved {n} days to {DATA}; mean={units.mean():.1f}, min={units.min()}, max={units.max()}")
