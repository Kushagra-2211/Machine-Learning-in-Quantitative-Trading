"""
ML pairs trading strategy #5: Gradient Boosting (XGBoost).

Added after the first four models (Random Forest, SVM, Neural Net, LSTM)
showed the two gradient-trained neural models underperforming on the
trading period despite reasonable formation-period fit (see ml_neural_net.py
/ ml_lstm.py docstrings for that diagnosis). Gradient-boosted trees are
consistently among the strongest performers in the broader
ML-on-small-tabular-financial-data literature -- often beating both plain
feedforward nets and even a single Random Forest -- so this is a natural,
well-regarded model to add to the comparison rather than concluding "ML
doesn't help here" from only the two weakest-performing model classes.

Same features/target/trading rule as every other model here (ml_common.py)
-- only the model differs.

NOTE: requires xgboost (`pip install xgboost`). I could not test-run this
file myself -- same situation as ml_neural_net.py / ml_lstm.py (sandbox
network restrictions). Please run it yourself first; wire it into main.py
the same way PyTorch is guarded (see TORCH_AVAILABLE / XGBOOST_AVAILABLE).
"""

import numpy as np
import pandas as pd
import xgboost as xgb

import config
import ml_common

N_ESTIMATORS = 300           # ceiling -- native early stopping (below) picks the actual best round
EARLY_STOPPING_ROUNDS = 20
VALIDATION_FRACTION = 0.2    # same chronological formation-period holdout used for NN/LSTM early stopping


def train_model(train_X, train_y, random_state=None, validation_fraction=VALIDATION_FRACTION,
                 n_estimators=N_ESTIMATORS, early_stopping_rounds=EARLY_STOPPING_ROUNDS):
    random_state = random_state if random_state is not None else config.RANDOM_SEED

    # Same chronological train/validation split used for the NN/LSTM's early
    # stopping (ml_common.train_validation_split_flat) -- XGBoost has its own
    # native early stopping support, so we hand it the validation slice
    # directly instead of guessing a fixed number of trees.
    X_tr, y_tr, X_val, y_val = ml_common.train_validation_split_flat(train_X, train_y, validation_fraction)

    # ===================== GRADIENT BOOSTING MODEL =====================
    # Shallow trees (max_depth=3) with subsampling and an L2 penalty
    # (reg_lambda) -- standard regularized-boosting defaults for a small
    # (~1,000-1,300 row) tabular formation set, deliberately conservative
    # rather than a deep/high-variance configuration.
    # ======================================================================
    model = xgb.XGBRegressor(
        n_estimators=n_estimators,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=random_state,
        early_stopping_rounds=early_stopping_rounds,
        eval_metric="rmse",
    )
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    print(f"  XGBoost: best iteration {model.best_iteration}/{n_estimators} "
          f"(best val RMSE {model.best_score:.6f}).")
    return model


def run_strategy(full_price1, full_price2, formation_end=None, trading_start=None, trading_end=None):
    data = ml_common.prepare_data(full_price1, full_price2, formation_end, trading_start, trading_end)

    # Guard against a walk-forward window where this pair's trading-period
    # features are entirely NaN (data gap in one leg) -- see
    # ml_common.trading_features_usable()'s docstring.
    if not ml_common.trading_features_usable(data["trading_features"]):
        predicted_spread = ml_common.no_signal_result("Gradient Boosting")
        result = ml_common.finish_strategy(
            full_price1, full_price2, predicted_spread, data["full_spread"],
            data["trading_start"], data["trading_end"],
        )
        result["model"] = None
        return result

    # Trees split on raw thresholds, so scaling isn't required the way it is
    # for SVM/NN/LSTM -- kept unscaled, same as Random Forest.
    trading_X = data["trading_features"].dropna()

    model = train_model(data["train_X"], data["train_y"])
    predicted_spread = model.predict(trading_X)
    predicted_spread = pd.Series(predicted_spread, index=trading_X.index)

    result = ml_common.finish_strategy(
        full_price1, full_price2, predicted_spread, data["full_spread"],
        data["trading_start"], data["trading_end"],
    )
    result["model"] = model
    return result


if __name__ == "__main__":
    import data_utils
    prices = data_utils.load_wide_prices()
    permnos = data_utils.get_sector_permnos()
    p1, p2 = permnos[0], permnos[1]
    result = run_strategy(prices[p1], prices[p2])
    cumret = (1 + result["daily_returns"]).cumprod()
    print(f"Gradient Boosting strategy smoke test on PERMNO {p1}/{p2}: "
          f"terminal cumulative return = {cumret.iloc[-1]:.4f}")
