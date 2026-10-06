# 🚀 Usage & Judge Evaluation (Google Colab)

Open a blank Google Colab Notebook (https://colab.research.google.com/) and run this all-in-one cell to download the code, train the LightGBM Quantile models, run the inventory cost simulation in ₹, and expose the FastAPI Swagger UI to the web via Cloudflare Tunnel:
```
# 1. Clone Repo & Install Dependencies
!rm -rf /content/AWS-Retail-management
!git clone https://github.com/kushagrbhatia0315-dot/AWS-Retail-management.git
%cd /content/AWS-Retail-management
!pip install -r requirements.txt

# 2. Run the Forecasting & Inventory Cost Pipeline
!python run_pipeline.py

# 3. Setup Cloudflare Tunnel (NO PASSWORD / TOKENS NEEDED)
!wget -q -nc https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
!chmod +x cloudflared-linux-amd64

# 4. Start FastAPI Server in the Background
import os, time
os.system("nohup python -m uvicorn src.app:app --host 127.0.0.1 --port 8000 > /content/fastapi.log 2>&1 &")
time.sleep(3)

# 5. Launch Cloudflare Tunnel & Print Public Swagger UI Link
!nohup ./cloudflared-linux-amd64 tunnel --url http://127.0.0.1:8000 > /content/tunnel.txt 2>&1 &
time.sleep(5)

print("\n" + "=" * 55)
print("🔗 CLICK BELOW TO VIEW THE INTERACTIVE API & SWAGGER DOCS:")
!grep -o 'https://.*\.trycloudflare.com' /content/tunnel.txt | head -n 1 | awk '{print $1 "/docs"}'
print("=" * 55 + "\n")
```
---

# AWS Retail Management

AWS Retail Management is a demand forecasting and inventory replenishment pipeline that produces calibrated prediction intervals (10th, 50th, and 90th percentiles) using Quantile LightGBM and evaluates financial inventory risk in Indian Rupees (₹).

## What It Includes

- Leak-free feature engineering past operational lead times (lags, rolling stats, cyclical calendar)
- Quantile LightGBM models trained on pinball loss for 10th, 50th, and 90th percentiles
- Monotonic interval enforcement to guarantee non-crossing bounds
- Baseline benchmark against Gaussian residual intervals
- Inventory cost simulation evaluating stockout vs. holding trade-offs in ₹
- Overconfidence diagnostics identifying interval breakdown during demand spikes
- FastAPI service serving prediction intervals and financial evaluation via JSON
- Automatic synthetic data generator fallback when no raw data is present
- Cloudflare Tunnel integration for zero-config public demo hosting in Google Colab
- Unit tests validating lag leakage prevention and inventory math accuracy

## Project Layout

```text
├── data/
│   ├── processed/      Cached feature artifacts
│   └── raw/            Raw sales datasets (train.csv)
├── docs/
│   └── writeup.md      Approach, decisions, and analysis write-up
├── notebooks/
│   └── 01_demand_forecasting_uq.ipynb
├── src/
│   ├── __init__.py
│   ├── app.py          FastAPI web service
│   ├── config.py       Cost constants (₹), paths, and parameters
│   ├── evaluation.py   PICP calibration, pinball loss, MAE, and RMSE
│   ├── features.py     Leak-free feature pipeline
│   ├── inventory.py    Asymmetric inventory cost simulator
│   └── models.py       Quantile LightGBM & monotonic interval projection
├── tests/
│   ├── __init__.py
│   ├── test_features.py   Data leakage test
│   └── test_inventory.py  Inventory math and currency test
├── .gitignore
├── requirements.txt
├── run_pipeline.py     End-to-end execution pipeline
└── README.md
```
## Requirements

- Python 3.10 or newer
- Terminal or Command Prompt / PowerShell

## Setup

Run these commands from the project root:

### 1. Create and activate the virtual environment

macOS/Linux:
python3 -m venv .venv
source .venv/bin/activate

PowerShell:
python -m venv .venv
.\.venv\Scripts\Activate.ps1

If PowerShell blocks activation:
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned

### 2. Install dependencies

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

Make sure your requirements.txt contains:
numpy
pandas
scikit-learn
lightgbm
fastapi
uvicorn
pytest

## Dataset Configuration & Exact Kaggle Source

### Automatic Synthetic Fallback (Zero Setup)

If data/raw/train.csv is not present, run_pipeline.py automatically generates a multi-store, multi-item retail sales history containing trends, day-of-week seasonality, Poisson noise, and promotional spikes. You do not need to download external data to run the pipeline out-of-the-box.

### Exact Kaggle Dataset: Store Item Demand Forecasting Challenge

The official dataset supported by this repository is the Kaggle competition:
- Competition Name: Store Item Demand Forecasting Challenge
- Kaggle URL: https://www.kaggle.com/c/demand-forecasting-kernels-only/data
- Target File: train.csv (5 years of daily store-item sales records)

Download and place the file using either method:

Method 1: Kaggle CLI
pip install kaggle
kaggle competitions download -c demand-forecasting-kernels-only
unzip demand-forecasting-kernels-only.zip -d data/raw/
unzip data/raw/train.csv.zip -d data/raw/

Method 2: Manual Download
1. Visit https://www.kaggle.com/c/demand-forecasting-kernels-only/data
2. Download train.csv
3. Run the following terminal commands to move it into place:
mkdir -p data/raw
mv ~/Downloads/train.csv data/raw/train.csv

## Run the Application

Run commands from the project root.

### 1. Run the ML Pipeline

Trains the quantile regressors, evaluates interval calibration, and runs the cost comparison:
python run_pipeline.py

### 2. Run the FastAPI Server

Launch the web API to serve pipeline outputs and decision metrics via JSON:
python -m uvicorn src.app:app --reload --host 127.0.0.1 --port 8000

URLs:
- API Health: http://127.0.0.1:8000/
- API documentation: http://127.0.0.1:8000/docs
- Full JSON metrics: http://127.0.0.1:8000/pipeline/all
- Inventory decision costs: http://127.0.0.1:8000/pipeline/inventory-costs

### 3. Run Automated Tests

Run the test suite to verify leak-free lag engineering and cost calculations:
pytest tests/ -v

## Inventory Cost Rules

Inventory replenishment costs are calculated with asymmetric penalties:

- Stockout unit cost (cu): ₹12.00 per unfulfilled unit
- Holding unit cost (co): ₹1.33 per excess unit

Because stockouts are significantly more expensive than holding surplus (cu >> co), ordering at the median (q50) produces an approximate 50% stockout risk. Ordering using the 90th percentile buffer (q90) provides the lowest total financial cost.

## API Summary

| Method | Path | Purpose |
| --- | --- | --- |
| GET | / | Health check confirmation |
| GET | /pipeline/all | Full pipeline metrics (calibration, costs, diagnostics) |
| GET | /pipeline/inventory-costs | Replenishment policy cost comparison in ₹ |
| GET | /pipeline/calibration | PICP empirical coverage and pinball loss |
| GET | /pipeline/diagnostics | Overconfidence analysis during demand spikes |
| POST | /pipeline/re-run | Trigger on-demand model re-training and re-evaluation |

## Troubleshooting

### ModuleNotFoundError: No module named 'run_pipeline'

Ensure you run the FastAPI command with the project root as your working directory:
python -m uvicorn src.app:app --reload --host 127.0.0.1 --port 8000

Also verify that src/app.py includes the root directory in the Python path:
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

### Port 8000 already in use

Run uvicorn on another port:
python -m uvicorn src.app:app --reload --host 127.0.0.1 --port 8001

Then open http://127.0.0.1:8001/docs.

### No data/raw/train.csv found

This is standard behavior when running out-of-the-box. The pipeline automatically switches to synthetic data generation so training completes successfully. To train on real data, place train.csv inside data/raw/.

## Git Workflow

To commit and push all modifications to GitHub:
git add .
git commit -m "docs: add Colab judge evaluation cell with Cloudflare tunnel and API docs"