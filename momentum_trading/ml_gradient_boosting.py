"""
ML momentum strategy #5: Gradient Boosting (XGBoost).

Same features/target/trading rule as every other model here (ml_common.py)
-- only the model differs. Added for the same reason
pairs_trading/ml_gradient_boosting.py was: gradient-boosted trees are
consistently strong on small tabular financial data, a natural addition
alongside Random Forest/SVM/Neural Net/LSTM rather than concluding "ML
doesn't help" from only a subset of model classes.

NOTE: requires xgboost (`pip install xgboost`).
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

    X_tr, y_tr, X_val, y_val = ml_common.train_validation_split_flat(train_X, train_y, validation_fraction)

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


def run_strategy(full_price, formation_end=None, trading_start=None, trading_end=None,
                  include_regime_features=False):
    data = ml_common.prepare_data(full_price, formation_end, trading_start, trading_end,
                                   include_regime_features=include_regime_features)

    # Trees split on raw thresholds, so scaling isn't required -- kept
    # unscaled, same as Random Forest.
    trading_X = data["trading_features"].dropna()

    model = train_model(data["train_X"], data["train_y"])
    predicted_return = pd.Series(model.predict(trading_X), index=trading_X.index)

    result = ml_common.finish_strategy_auto(
        predicted_return, data["full_return"], data["trading_start"], data["trading_end"], data["train_y"],
    )
    result["model"] = model
    return result


if __name__ == "__main__":
    import data_utils
    prices = data_utils.load_wide_prices()
    coin = config.COINS[0]
    result = run_strategy(prices[coin])
    cumret = (1 + result["daily_returns"]).cumprod()
    print(f"Gradient Boosting strategy smoke test on {coin}: "
          f"terminal cumulative return = {cumret.iloc[-1]:.4f}")
