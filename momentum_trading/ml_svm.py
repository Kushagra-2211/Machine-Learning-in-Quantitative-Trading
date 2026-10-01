"""
ML momentum strategy #2: Support Vector Regression (SVM).

Same setup as ml_random_forest.py (predicts tomorrow's return from the
features in ml_common.py, same trading rule) -- only the model differs.
Linear-kernel SVR, matching the choice pairs_trading/ml_svm.py made per Peng
Liu Ch.10.
"""

import pandas as pd
from sklearn.svm import SVR

import config
import ml_common


def train_model(train_X, train_y, kernel="linear"):
    model = SVR(kernel=kernel)
    model.fit(train_X, train_y)
    return model


def run_strategy(full_price, formation_end=None, trading_start=None, trading_end=None, kernel="linear",
                  include_regime_features=False):
    data = ml_common.prepare_data(full_price, formation_end, trading_start, trading_end,
                                   include_regime_features=include_regime_features)

    # SVM's margin/regularization is sensitive to input scale -- standardize
    # using formation-period statistics only.
    trading_X = data["trading_features"].dropna()
    train_X_scaled, (trading_X_scaled,), _ = ml_common.standardize_features(data["train_X"], [trading_X])

    model = train_model(train_X_scaled, data["train_y"], kernel=kernel)
    predicted_return = pd.Series(model.predict(trading_X_scaled), index=trading_X_scaled.index)

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
    print(f"SVM strategy smoke test on {coin}: "
          f"terminal cumulative return = {cumret.iloc[-1]:.4f}")
