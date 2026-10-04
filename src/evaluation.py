import numpy as np
import pandas as pd
def pinball_loss(y_true, y_pred, alpha):
    residual = y_true - y_pred
    return np.mean(np.maximum(alpha * residual, (alpha - 1) * residual))
def evaluate_intervals(y_true, q10, q50, q90):
    mae = float(np.mean(np.abs(y_true - q50)))
    rmse = float(np.sqrt(np.mean((y_true - q50) ** 2)))
    wape = float(np.sum(np.abs(y_true - q50)) / np.sum(y_true))
    covered = (y_true >= q10) & (y_true <= q90)
    picp = float(np.mean(covered)) 
    mean_width = float(np.mean(q90 - q10))
    mean_pinball = float(
        (pinball_loss(y_true, q10, 0.10) + 
         pinball_loss(y_true, q50, 0.50) + 
         pinball_loss(y_true, q90, 0.90)) / 3.0
    )
    return {
        "MAE": round(mae, 3),
        "RMSE": round(rmse, 3),
        "WAPE": round(wape, 4),
        "Coverage (PICP, target 0.80)": round(picp, 4),
        "Mean Interval Width": round(mean_width, 2),
        "Mean Pinball Loss": round(mean_pinball, 4)
    }
def analyze_overconfidence(test_df: pd.DataFrame, y_true: np.ndarray, q10: np.ndarray, q90: np.ndarray):
    """Identifies subsets where model coverage drops (intervals are too narrow)."""
    df = test_df.copy()
    df["covered"] = (y_true >= q10) & (y_true <= q90)
    df["is_demand_spike"] = y_true > (df["rolling_mean_28"] + 2 * df["rolling_std_28"])
    spike_coverage = float(df.groupby("is_demand_spike")["covered"].mean().get(True, 0.0))
    normal_coverage = float(df.groupby("is_demand_spike")["covered"].mean().get(False, 0.0))
    dow_coverage = df.groupby("dayofweek")["covered"].mean().to_dict()
    return {
        "Normal Demand Coverage": round(normal_coverage, 4),
        "Demand Spike Coverage (Overconfidence Indicator)": round(spike_coverage, 4),
        "Day of Week Coverage": {k: round(v, 4) for k, v in dow_coverage.items()}}