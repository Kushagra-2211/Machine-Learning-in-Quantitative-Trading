"""
ML pairs trading strategy #1: Random Forest.

Predicts TOMORROW's spread from six lagged features (see ml_common.py), then
trades on how far that prediction sits from the recent historical average --
same z-score/entry/exit rule as the classical strategy (signals.py), so the
only thing that differs from classical_strategy.py is the mechanism
generating the signal.

Configuration follows Peng Liu, "Quantitative Trading Strategies Using
Python" (2023), Ch.10: sklearn RandomForestRegressor, n_estimators=100.
See DISSERTATION_PLAN.md and ml_common.py's docstring for the deliberate
departures from the book (lagged features, shared spread definition, extra
SpreadLag1 feature).
"""

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

import config
import ml_common


def train_model(train_X, train_y, random_state=None):
    random_state = random_state if random_state is not None else config.RANDOM_SEED
    # ===================== RANDOM FOREST MODEL =====================
    # 100 trees, trained on the formation period only. This is the one
    # model that gets swapped out across ml_random_forest.py / ml_svm.py /
    # ml_neural_net.py / ml_lstm.py -- everything else (features, target,
    # trading rule) is identical across all four.
    # =================================================================
    model = RandomForestRegressor(n_estimators=100, random_state=random_state)
    model.fit(train_X, train_y)
    return model


def run_strategy(full_price1, full_price2, formation_end=None, trading_start=None, trading_end=None):
    data = ml_common.prepare_data(full_price1, full_price2, formation_end, trading_start, trading_end)

    # Guard against a walk-forward window where this pair's trading-period
    # features are entirely NaN (data gap in one leg) -- see
    # ml_common.trading_features_usable()'s docstring.
    if not ml_common.trading_features_usable(data["trading_features"]):
        predicted_spread = ml_common.no_signal_result("Random Forest")
        result = ml_common.finish_strategy(
            full_price1, full_price2, predicted_spread, data["full_spread"],
            data["trading_start"], data["trading_end"],
        )
        result["model"] = None
        return result

    model = train_model(data["train_X"], data["train_y"])
    predicted_spread = model.predict(data["trading_features"].dropna())
    predicted_spread = pd.Series(predicted_spread, index=data["trading_features"].dropna().index)

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
    print(f"Random Forest strategy smoke test on PERMNO {p1}/{p2}: "
          f"terminal cumulative return = {cumret.iloc[-1]:.4f}")
