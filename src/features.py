import numpy as np
import pandas as pd

def build_features(df: pd.DataFrame, target_col: str = "sales", lead_time: int = 7) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["store", "item", "date"]).reset_index(drop=True)
    df["dayofweek"] = df["date"].dt.dayofweek
    df["day"] = df["date"].dt.day
    df["month"] = df["date"].dt.month
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)
    df["sin_dow"] = np.sin(2 * np.pi * df["dayofweek"] / 7.0)
    df["cos_dow"] = np.cos(2 * np.pi * df["dayofweek"] / 7.0)
    df["sin_month"] = np.sin(2 * np.pi * df["month"] / 12.0)
    df["cos_month"] = np.cos(2 * np.pi * df["month"] / 12.0)
    grouped = df.groupby(["store", "item"])[target_col]
    for lag in [lead_time, lead_time + 1, lead_time + 7, lead_time + 14, lead_time + 28]:
        df[f"lag_{lag}"] = grouped.shift(lag)
    shifted = grouped.shift(lead_time)
    for window in [7, 14, 28, 60]:
        df[f"rolling_mean_{window}"] = shifted.transform(lambda x: x.rolling(window, min_periods=3).mean())
        df[f"rolling_std_{window}"] = shifted.transform(lambda x: x.rolling(window, min_periods=3).std())
        df[f"rolling_max_{window}"] = shifted.transform(lambda x: x.rolling(window, min_periods=3).max())
    df["volatility_ratio_7_28"] = df["rolling_std_7"] / (df["rolling_std_28"] + 1e-5)
    return df