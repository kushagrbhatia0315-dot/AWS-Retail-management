import pandas as pd
import numpy as np
from src.features import build_features
def test_lag_leakage_prevention():
    dates = pd.date_range("2024-01-01", periods=30, freq="D")
    df = pd.DataFrame({"date": dates, "store": 1, "item": 1, "sales": np.arange(30)})
    feat_df = build_features(df, target_col="sales", lead_time=7)
    assert feat_df.loc[7, "lag_7"] == df.loc[0, "sales"]