# AWS Retail Management: Demand Forecasting with Uncertainty Quantification

Probabilistic retail demand forecasting pipeline that produces calibrated prediction intervals (10th, 50th, and 90th quantiles) to balance asymmetric stockout and holding costs in Indian Rupees (₹).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688.svg?logo=fastapi&logoColor=white)](http://127.0.0.1:8000/docs)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)

---

## Overview

Single point forecasts hide operational risk. Overstocking costs inventory holding fees ($c_o$), while stockouts forfeit customer goodwill and direct revenue ($c_u$). When $c_u \gg c_o$, optimal replenishment must target higher quantiles rather than simple expected averages.

This repository implements:
- **Feature Engineering:** Leak-free rolling aggregations, lag features past operational lead times, and cyclical calendar dynamics.
- **Quantile Gradient Boosting:** LightGBM regressors trained under pinball/quantile loss ($\alpha \in \{0.10, 0.50, 0.90\}$) with post-hoc monotonic sorting.
- **Baseline Benchmarking:** Standard MSE point forecast with Gaussian residual prediction bands.
- **Financial Inventory Simulation:** Asymmetric stockout vs. holding cost trade-offs evaluated in Indian Rupees (₹).
- **Overconfidence Diagnostics:** Automated detection of interval breakdown during demand spikes.
- **FastAPI JSON Engine:** Ready-to-serve REST API exposing calibration and financial simulation outputs with full Swagger documentation.

---

## Project Structure

```text
├── data/
│   ├── processed/      # Cached, feature-engineered artifacts
│   └── raw/            # Raw sales data (train.csv)
├── docs/
│   └── writeup.md      # Methodological details & decision analysis
├── notebooks/
│   └── 01_demand_forecasting_uq.ipynb
├── src/
│   ├── app.py          # FastAPI web service
│   ├── config.py       # Cost constants (₹) and global hyperparams
│   ├── evaluation.py   # PICP calibration & pinball loss calculations
│   ├── features.py     # Leak-free time-series transformations
│   ├── inventory.py    # Cost simulator for holding vs stockout risk
│   └── models.py       # Quantile LightGBM & monotonic interval enforcement
├── tests/
│   ├── test_features.py
│   └── test_inventory.py
├── README.md
├── requirements.txt
└── run_pipeline.py     # Execution entry point