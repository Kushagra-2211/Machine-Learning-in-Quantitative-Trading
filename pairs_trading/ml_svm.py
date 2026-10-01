"""
ML pairs trading strategy #2: Support Vector Regression (SVM).

Same setup as ml_random_forest.py (predicts tomorrow's spread from the six
lagged features in ml_common.py, same trading rule) -- only the model
differs.

Configuration follows Peng Liu, "Quantitative Trading Strategies Using
Python" (2023), Ch.10: sklearn SVR with a linear kernel. In the book's own
results, this was the only one of the three models (SVM/RF/NN) that made
money in the actual trading test, despite having worse prediction RMSE than
Random Forest -- worth discussing directly in the Results section.

Note: SVR has no random_state parameter (unlike RandomForestRegressor /
sklearn's MLPRegressor) -- a linear-kernel SVR's solution is deterministic
given the same data, so there's nothing to seed.
"""

import pandas as pd
from sklearn.svm import SVR

import config
import ml_common


def train_model(train_X, train_y, kernel="linear"):
    # ===================== SVM (SUPPORT VECTOR REGRESSION) =====================
    # Linear-kernel SVR, matching Peng Liu Ch.10. Trained on the formation
    # period only, same features/target as every other ML model here.
    # =============================================================================
    model = SVR(kernel=kernel)
    model.fit(train_X, train_y)
    return model


def run_strategy(full_price1, full_price2, formation_end=None, trading_start=None, trading_end=None, kernel="linear"):
    data = ml_common.prepare_data(full_price1, full_price2, formation_end, trading_start, trading_end)

    # Guard against a walk-forward window where this pair's trading-period
    # features are entirely NaN (data gap in one leg) -- see
    # ml_common.trading_features_usable()'s docstring. Must be checked
    # BEFORE standardize_features(), since StandardScaler.transform() also
    # raises on a zero-row input, not just model.predict().
    if not ml_common.trading_features_usable(data["trading_features"]):
        predicted_spread = ml_common.no_signal_result("SVM")
        result = ml_common.finish_strategy(
            full_price1, full_price2, predicted_spread, data["full_spread"],
            data["trading_start"], data["trading_end"],
        )
        result["model"] = None
        return result

    # SVM's margin/regularization is sensitive to input scale -- standardize
    # using formation-period statistics only (see ml_common.standardize_features).
    trading_X = data["trading_features"].dropna()
    train_X_scaled, (trading_X_scaled,), _ = ml_common.standardize_features(data["train_X"], [trading_X])

    model = train_model(train_X_scaled, data["train_y"], kernel=kernel)
    predicted_spread = model.predict(trading_X_scaled)
    predicted_spread = pd.Series(predicted_spread, index=trading_X_scaled.index)

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
    print(f"SVM strategy smoke test on PERMNO {p1}/{p2}: "
          f"terminal cumulative return = {cumret.iloc[-1]:.4f}")
