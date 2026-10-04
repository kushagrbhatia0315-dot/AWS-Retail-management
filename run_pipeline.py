import pandas as pd
import numpy as np
from src.config import DATA_RAW, LEAD_TIME, QUANTILES, UNIT_STOCKOUT_COST, UNIT_HOLDING_COST
from src.features import build_features
from src.models import train_quantile_models, train_point_baseline, enforce_monotonic_quantiles
from src.evaluation import evaluate_intervals, analyze_overconfidence
from src.inventory import simulate_inventory_costs
def load_or_generate_data():
    csv_file = DATA_RAW / "train.csv"
    if csv_file.exists():
        print(f"[INFO] Ingesting dataset from {csv_file}...")
        df = pd.read_csv(csv_file)
        if "sales" not in df.columns and "demand" in df.columns:
            df = df.rename(columns={"demand": "sales"})
        return df
    print("[INFO] No data/raw/train.csv found. Generating synthetic multi-store retail data...")
    dates = pd.date_range("2022-01-01", "2024-12-31", freq="D")
    records = []
    np.random.seed(42)
    for store in range(1, 4):
        for item in range(1, 6):
            base = np.random.uniform(25, 45)
            trend = np.linspace(0, 10, len(dates))
            dow_season = np.sin(2 * np.pi * dates.dayofweek / 7.0) * 8
            promo_spikes = (np.random.rand(len(dates)) > 0.97) * np.random.uniform(20, 40, size=len(dates))
            noise = np.random.poisson(lam=4, size=len(dates))
            sales = np.maximum(0, base + trend + dow_season + promo_spikes + noise).astype(int)
            for d, s in zip(dates, sales):
                records.append({"date": d, "store": store, "item": item, "sales": s})
    return pd.DataFrame(records)
def main():
    raw_df = load_or_generate_data()
    print("[INFO] Building leak-free features...")
    df = build_features(raw_df, target_col="sales", lead_time=LEAD_TIME)
    df = df.dropna().reset_index(drop=True)
    train_end = "2024-04-01"
    val_end = "2024-08-01"
    train_df = df[df["date"] < train_end]
    val_df = df[(df["date"] >= train_end) & (df["date"] < val_end)]
    test_df = df[df["date"] >= val_end].copy()
    drop_cols = ["date", "store", "item", "sales"]
    feature_cols = [c for c in df.columns if c not in drop_cols]
    print("[INFO] Training LightGBM Quantile Regressors (q10, q50, q90)...")
    q_models = train_quantile_models(train_df, train_df["sales"], val_df, val_df["sales"], feature_cols, QUANTILES)
    raw_q10 = q_models[0.10].predict(test_df[feature_cols])
    raw_q50 = q_models[0.50].predict(test_df[feature_cols])
    raw_q90 = q_models[0.90].predict(test_df[feature_cols])
    q10, q50, q90 = enforce_monotonic_quantiles(raw_q10, raw_q50, raw_q90)
    print("[INFO] Training Baseline Point Model (MSE) with Residual Intervals...")
    point_model = train_point_baseline(train_df, train_df["sales"], feature_cols)
    val_resid = val_df["sales"] - point_model.predict(val_df[feature_cols])
    sigma_val = np.std(val_resid)
    test_preds = point_model.predict(test_df[feature_cols])
    base_q10 = np.clip(test_preds - 1.282 * sigma_val, 0, None)
    base_q90 = np.clip(test_preds + 1.282 * sigma_val, 0, None)
    y_test = test_df["sales"].values
    print("\n" + "=" * 25 + " 1. INTERVAL CALIBRATION " + "=" * 25)
    print("Quantile LightGBM: ", evaluate_intervals(y_test, q10, q50, q90))
    print("Gaussian Baseline: ", evaluate_intervals(y_test, base_q10, test_preds, base_q90))
    print("\n" + "=" * 25 + " 2. INVENTORY DECISION COSTS " + "=" * 25)
    cost_point = simulate_inventory_costs(y_test, q50, UNIT_STOCKOUT_COST, UNIT_HOLDING_COST)
    cost_gauss = simulate_inventory_costs(y_test, base_q90, UNIT_STOCKOUT_COST, UNIT_HOLDING_COST)
    cost_q90 = simulate_inventory_costs(y_test, q90, UNIT_STOCKOUT_COST, UNIT_HOLDING_COST)
    print("Policy 1: Point Forecast (Median / q50):  ", cost_point)
    print("Policy 2: Gaussian Residual Safety Buffer:", cost_gauss)
    print("Policy 3: Quantile q90 Buffer Policy:    ", cost_q90)
    print("\n" + "=" * 25 + " 3. OVERCONFIDENCE DIAGNOSTICS " + "=" * 25)
    print(analyze_overconfidence(test_df, y_test, q10, q90))
if __name__ == "__main__":
    main()