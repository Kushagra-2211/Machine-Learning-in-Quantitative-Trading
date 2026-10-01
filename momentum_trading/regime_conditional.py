"""
Regime-conditional training (experiment): instead of one model trained on
the whole formation window, train TWO separate models per walk-forward
period -- one on formation days classified "Trending", one on formation
days classified "Range-bound" -- and predict each trading day using
whichever model matches THAT day's own regime.

Only implemented for the four FLAT-feature models (Random Forest, SVM,
Neural Net, Gradient Boosting), which all share ml_common.prepare_data()'s
data shape and already expose a (train_X, train_y) -> model train_model()
plus a model.predict(X)-shaped (or ml_neural_net.predict(model, X)-shaped)
prediction step -- LSTM's sequence-windowed data (each row is a whole
(seq_len, n_features) window rather than a single day) doesn't split into
"days classified Trending vs Range-bound" nearly as cleanly, so it's
excluded from this experiment (see DISSERTATION_PLAN.md for the explicit
scope note).

CAUSALITY: the Trending/Range-bound threshold used to split BOTH the
formation data (for training) and the trading data (for dispatching
predictions) is the MEDIAN Kaufman Efficiency Ratio computed from the
FORMATION PERIOD ONLY, fixed once per walk-forward period before any
trading-period data is touched -- this is different from (and more
conservative than) main.py's own regime-conditional REPORTING, which
classifies regime using a median computed over the whole trading period
with the benefit of hindsight (fine for descriptive analysis after the
fact, not fine as an actual trading rule). Here, the threshold is knowable
in real time at the start of each walk-forward period, so this is a
genuinely tradeable design, not just a hindsight split.
"""

import numpy as np
import pandas as pd

import config
import ml_common
import regime

MIN_REGIME_ROWS = 30  # if either regime bucket has fewer formation rows than this, fall back to one pooled model


def run_regime_conditional(train_fn, predict_fn, needs_scaling, full_price,
                            formation_end=None, trading_start=None, trading_end=None,
                            short_window=None, long_window=None):
    """Generic regime-conditional wrapper for any flat-feature model.

    train_fn(X, y) -> model
    predict_fn(model, X) -> np.array of predictions
    needs_scaling: whether to standardize features on formation stats
    before training (True for SVM/Neural Net, False for tree models)."""
    formation_end = formation_end or config.FORMATION_END
    trading_start = trading_start or config.TRADING_START
    trading_end = trading_end or config.TRADING_END

    full_return = full_price.pct_change()
    full_features = ml_common.build_features(full_price, short_window, long_window)
    full_target = ml_common.build_target(full_price)
    er = regime.efficiency_ratio(full_price, window=config.REGIME_WINDOW)

    formation_features = full_features.loc[:formation_end]
    formation_target = full_target.loc[:formation_end]
    formation_er = er.loc[:formation_end]

    train_data = pd.concat(
        [formation_features, formation_target.rename("target"), formation_er.rename("er")], axis=1
    ).dropna()
    X_cols = list(formation_features.columns)

    threshold = train_data["er"].median()  # FIXED, computed from formation period only -- causal

    trading_features_raw = full_features.loc[trading_start:trading_end].dropna()
    trading_er = er.reindex(trading_features_raw.index)

    if needs_scaling:
        train_X_scaled, (trading_X_scaled,), _ = ml_common.standardize_features(
            train_data[X_cols], [trading_features_raw]
        )
        train_source = train_data.copy()
        train_source[X_cols] = train_X_scaled.values
        trading_X_for_predict = trading_X_scaled
    else:
        train_source = train_data
        trading_X_for_predict = trading_features_raw

    trending_train = train_source[train_source["er"] >= threshold]
    rangebound_train = train_source[train_source["er"] < threshold]

    if len(trending_train) < MIN_REGIME_ROWS or len(rangebound_train) < MIN_REGIME_ROWS:
        # Not enough formation data in one of the two buckets this period
        # (can happen in the very first walk-forward period, or a coin with
        # unusually persistent trendiness/choppiness) -- fall back to ONE
        # pooled model for both regimes rather than training on too little
        # data, same fallback philosophy as elsewhere in this codebase.
        print(f"  [regime-conditional] formation split too small "
              f"(Trending={len(trending_train)}, Range-bound={len(rangebound_train)}) "
              f"-- falling back to a single pooled model for this period.")
        pooled_model = train_fn(train_source[X_cols], train_source["target"])
        model_trending = model_rangebound = pooled_model
    else:
        model_trending = train_fn(trending_train[X_cols], trending_train["target"])
        model_rangebound = train_fn(rangebound_train[X_cols], rangebound_train["target"])

    pred_trending = pd.Series(predict_fn(model_trending, trading_X_for_predict), index=trading_X_for_predict.index)
    pred_rangebound = pd.Series(predict_fn(model_rangebound, trading_X_for_predict), index=trading_X_for_predict.index)

    is_trending_day = (trading_er >= threshold).reindex(trading_X_for_predict.index).fillna(False)
    predicted_return = pred_trending.where(is_trending_day, pred_rangebound)

    train_y_for_sizing = train_source["target"]  # whole formation target, for POSITION_SIZING's scale if that's also on
    result = ml_common.finish_strategy_auto(
        predicted_return, full_return, trading_start, trading_end, train_y_for_sizing,
    )
    result["model"] = (model_trending, model_rangebound)
    result["threshold"] = threshold
    return result


# ===================== per-model wrappers =====================
# Each matches the (price, formation_end=, trading_start=, trading_end=)
# signature every other run_strategy() in this package uses, so these can be
# swapped in for the normal ones in main.py's run_all_strategies_for_coin()
# via config.REGIME_CONDITIONAL_TRAINING -- see maybe_walk_forward() there.
# Imported lazily inside each wrapper (not at module level) so this file can
# still be imported even if a strategy module fails to import (e.g. no
# PyTorch) -- mirrors main.py's own TORCH_AVAILABLE/XGBOOST_AVAILABLE guards.
def run_random_forest(full_price, formation_end=None, trading_start=None, trading_end=None, **kwargs):
    import ml_random_forest
    return run_regime_conditional(
        ml_random_forest.train_model, lambda model, X: model.predict(X), needs_scaling=False,
        full_price=full_price, formation_end=formation_end, trading_start=trading_start, trading_end=trading_end,
    )


def run_svm(full_price, formation_end=None, trading_start=None, trading_end=None, **kwargs):
    import ml_svm
    return run_regime_conditional(
        ml_svm.train_model, lambda model, X: model.predict(X), needs_scaling=True,
        full_price=full_price, formation_end=formation_end, trading_start=trading_start, trading_end=trading_end,
    )


def run_neural_net(full_price, formation_end=None, trading_start=None, trading_end=None, **kwargs):
    import ml_neural_net
    return run_regime_conditional(
        ml_neural_net.train_model, ml_neural_net.predict, needs_scaling=True,
        full_price=full_price, formation_end=formation_end, trading_start=trading_start, trading_end=trading_end,
    )


def run_gradient_boosting(full_price, formation_end=None, trading_start=None, trading_end=None, **kwargs):
    import ml_gradient_boosting
    return run_regime_conditional(
        ml_gradient_boosting.train_model, lambda model, X: model.predict(X), needs_scaling=False,
        full_price=full_price, formation_end=formation_end, trading_start=trading_start, trading_end=trading_end,
    )
