#!/usr/bin/env bash
# Reproduce every number and figure in the report (about 6 minutes on a laptop CPU).
set -e
cd "$(dirname "$0")/src"
python 00_generate_data.py   # simulated POS data -> data/
python 01_eda.py             # tasks 1-3
python 02_classical.py       # tasks 4-9
python 03_neural.py          # tasks 10, 11, 13
python 04_decision.py        # tasks 12, 14
python 05_summary.py         # comparison table and chart
