import numpy as np
import lightgbm as lgb
from typing import Dict
def train_quantile_models(
    X_train, y_train, X_val, y_val, feature_cols, quantiles=[0.10, 0.50, 0.90]
) -> Dict[float, lgb.LGBMRegressor]:
    models = {}
    for q in quantiles:
        params = {
            "objective": "quantile",
            "alpha": q,
            "metric": "quantile",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_child_samples": 20,
            "n_estimators": 400,
            "random_state": 42,
            "verbose": -1
        }
        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_train[feature_cols], y_train,
            eval_set=[(X_val[feature_cols], y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)]
        )
        models[q] = model
    return models
def train_point_baseline(X_train, y_train, feature_cols):
    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=400,
        learning_rate=0.05,
        random_state=42,
        verbose=-1)
    model.fit(X_train[feature_cols], y_train)
    return model
def enforce_monotonic_quantiles(q10: np.ndarray, q50: np.ndarray, q90: np.ndarray):
    """Guarantees physical bounds: 0 <= q10 <= q50 <= q90."""
    q10_clean = np.clip(q10, 0, None)
    q50_clean = np.maximum(q10_clean, np.clip(q50, 0, None))
    q90_clean = np.maximum(q50_clean, np.clip(q90, 0, None))
    return q10_clean, q50_clean, q90_clean