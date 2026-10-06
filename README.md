# AWS Retail Management: Demand Forecasting with Uncertainty Quantification (UQ)

A production-grade retail demand forecasting and inventory replenishment pipeline that moves beyond single-point predictions. This repository implements **probabilistic quantile gradient boosting** to produce well-calibrated prediction intervals (10th, 50th, and 90th percentiles), directly optimizing real-world inventory cost trade-offs under asymmetric loss in Indian Rupees (₹).

Includes an interactive **FastAPI** microservice serving inference diagnostics and decision analytics via JSON endpoints.

---

## Table of Contents
- [Business Problem & Objective](#business-problem--objective)
- [System Architecture & Methodology](#system-architecture--methodology)
- [Project Directory Structure](#project-directory-structure)
- [Quickstart: Running via Google Colab](#quickstart-running-via-google-colab-recommended)
- [Local Installation & Setup](#local-installation--setup)
- [Execution & Verification](#execution--verification)
- [FastAPI Microservice & Documentation](#fastapi-microservice--documentation)
- [Financial Cost & Calibration Analysis](#financial-cost--calibration-analysis)
- [Diagnostics: Overconfidence & Heteroscedasticity](#diagnostics-overconfidence--heteroscedasticity)

---

## Business Problem & Objective

Traditional supply chain forecasting models predict only the conditional mean or median:
$$\mathbb{E}[Y \vert{} X]$$

In retail inventory planning, uncertainty is non-symmetric:
* **Understocking (Stockouts):** Causes lost sales revenue, unfulfilled demand, and permanent customer churn. Unit cost: $c_u$.
* **Overstocking (Holding Costs):** Causes tied-up working capital, shelf-space consumption, and spoilage/depreciation. Unit cost: $c_o$.

When the cost of a stockout is substantially higher than the cost of holding excess stock ($c_u \gg c_o$), ordering at the point forecast ($q_{50}$) guarantees a ~50% stockout risk. 

**Core Objectives:**
1. Generate calibrated non-crossing prediction intervals ($q_{10}, q_{50}, q_{90}$) using LightGBM with Pinball/Quantile Loss.
2. Formulate an inventory safety-stock policy using the 90th percentile to minimize asymmetric financial losses (in ₹).
3. Benchmark against classical Gaussian residual bands ($1.282\sigma$).
4. Diagnose model overconfidence around demand spikes and promotional bursts.

---

## System Architecture & Methodology

```text
[ Raw Data / Kaggle / Synthetic ]
               │
               ▼
   [ Leak-Free Feature Pipeline ] ──> Lags (past lead-time L=7), Rolling Stats, Fourier/Cyclical Calendar
               │
               ▼
    [ Quantile LightGBM Models ] ──> α ∈ {0.10, 0.50, 0.90} via Pinball Loss
               │
               ▼
 [ Monotonic Boundary Enforcement ] ──> Clip negatives & guarantee: 0 ≤ q10 ≤ q50 ≤ q90
               │
               ▼
   [ Asymmetric Cost Simulator ] ──> Compute Stockout vs Holding Costs in ₹ (INR)
               │
               ▼
      [ FastAPI JSON Engine ] ────> REST Endpoints (/docs, /pipeline/all, /pipeline/inventory-costs)