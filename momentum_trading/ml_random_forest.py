"""
ML momentum strategy #1: Random Forest.

Predicts TOMORROW's simple return from the momentum features in
ml_common.py, then trades on the SIGN of that prediction -- same shift-then-
multiply trading rule as the classical strategy (signals.py), so the only
thing that differs from classical_strategy.py is the mechanism generating
the signal.

Configuration follows the same choice pairs_trading/ml_random_forest.py made
per Peng Liu Ch.10: sklearn RandomForestRegressor, n_estimators=100. (The
book itself has no ML+momentum chapter -- this ML layer is this
dissertation's own contribution, see DISSERTATION_PLAN.md.)
"""

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

import config
import ml_common


def train_model(train_X, train_y, random_state=None):
    random_state = random_state if random_state is not None else config.RANDOM_SEED
    model = RandomForestRegressor(n_estimators=100, random_state=random_state)
    model.fit(train_X, train_y)
    return model


def run_strategy(full_price, formation_end=None, trading_start=None, trading_end=None,
                  include_regime_features=False):
    data = ml_common.prepare_data(full_price, formation_end, trading_start, trading_end,
                                   include_regime_features=include_regime_features)

    model = train_model(data["train_X"], data["train_y"])
    trading_X = data["trading_features"].dropna()
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
    print(f"Random Forest strategy smoke test on {coin}: "
          f"terminal cumulative return = {cumret.iloc[-1]:.4f}")
