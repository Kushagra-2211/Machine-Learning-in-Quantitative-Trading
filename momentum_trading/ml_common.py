"""
Shared plumbing used by EVERY ML momentum variant (Random Forest, SVM,
Neural Network, LSTM, Gradient Boosting -- see ml_random_forest.py,
ml_svm.py, ml_neural_net.py, ml_lstm.py, ml_gradient_boosting.py). Keeping
this in one place means the comparison across models isolates just the
choice of model: features, target, the formation/trading split, and the
trading rule are all identical across the five, so any performance
difference is attributable to the model itself -- same design philosophy as
pairs_trading/ml_common.py.

TASK FRAMED AS: predicting continuation/reversal of trend (DISSERTATION_PLAN.md's
stated goal for the crypto momentum ML layer) -- operationalized as regression
on TOMORROW's simple return from each coin's own recent momentum/technical
features, with the trading signal taken as the SIGN of that prediction
(positive predicted return -> long, negative -> short). This is the direct
single-asset analog of the pairs strand's "predict tomorrow's spread, trade
on the sign of how far the prediction sits from its recent average" design,
simplified because there's no spread/hedge-ratio machinery here -- just one
asset's own price history.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import config
import classical_strategy
import signals
import regime


def compute_rsi(price, window=14):
    """===================== RELATIVE STRENGTH INDEX (RSI) =====================
    Standard 14-day RSI: average gain / average loss over the trailing
    window, rescaled to 0-100. RSI > 70 is conventionally read as
    "overbought", < 30 as "oversold" -- included here as a momentum-strength
    feature distinct from the raw SMA crossover the classical strategy
    trades on, giving the ML models a complementary read on trend strength
    rather than just trend direction.
    ================================================================================"""
    delta = price.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    # Where avg_loss is 0 (straight uptrend over the window), rs is inf and
    # the formula above already correctly gives rsi -> 100; guard only the
    # 0/0 case (both avg_gain and avg_loss are 0 -- a completely flat price).
    rsi = rsi.where(~((avg_gain == 0) & (avg_loss == 0)), 50.0)
    return rsi


def build_features(price, short_window=None, long_window=None, include_regime_features=False):
    """===================== FEATURE ENGINEERING (all ML models) =====================
    All computed using only information available up to and including day t
    -- this is the input every ML model (Random Forest, SVM, Neural Net,
    LSTM, Gradient Boosting) is trained on, so the SAME features feed all of
    them.

      - Return / ReturnLag1/2/3/5: today's and the last few days' simple
        returns -- short-horizon autocorrelation/momentum signal.
      - SMAShort / SMALong: the same two moving averages the classical
        strategy trades on (config.SHORT_WINDOW / config.LONG_WINDOW).
      - SMADiff: (SMAShort - SMALong) / SMALong -- a continuous-valued
        version of the classical strategy's binary crossover signal, letting
        the model use trend STRENGTH, not just trend direction.
      - Volatility20: rolling std of daily returns -- a regime-volatility
        signal (trend-following strategies are known to behave differently
        in high- vs low-volatility regimes).
      - ROC10: 10-day rate of change (price_t / price_{t-10} - 1) -- a
        classic momentum indicator, independent of the SMA pairing.
      - RSI14: see compute_rsi() above.
      - EfficiencyRatio: Kaufman's Efficiency Ratio (see regime.py),
        computed on the SAME rolling window (config.REGIME_WINDOW, causal --
        uses only the trailing `window` days, no look-ahead) main.py later
        uses to classify each trading day as Trending/Range-bound. Added as
        an EXPLICIT regime-strength feature so the ML models can, in
        principle, learn to weight the trend signal more heavily when the
        market is genuinely trending and hedge/back off when it's chopping
        sideways -- exactly the distinction the sanity-check found Classical
        exploiting (Sharpe 2.31 in Trending vs -2.51 in Range-bound) that
        the ML models, lacking any explicit regime information, could not.
      - ClassicalSignal: the classical strategy's own +1/-1 SMA-crossover
        signal (sign of SMADiff above), added as an explicit feature so the
        ML models can learn "trust the classical signal more/less" rather
        than needing to re-derive trend direction from the raw SMAs
        themselves. Both additions are OPTIONAL and OFF by default
        (include_regime_features=False) so every existing result/plot in
        results/ still reproduces exactly; pass include_regime_features=True
        to add them (see run_strategy() in each ml_*.py file).
    ==================================================================================="""
    short_window = short_window if short_window is not None else config.SHORT_WINDOW
    long_window = long_window if long_window is not None else config.LONG_WINDOW

    ret = price.pct_change()
    sma_short = price.rolling(short_window).mean()
    sma_long = price.rolling(long_window).mean()
    sma_diff = (sma_short - sma_long) / sma_long

    features = pd.DataFrame({
        "Return": ret,
        "ReturnLag1": ret.shift(1),
        "ReturnLag2": ret.shift(2),
        "ReturnLag3": ret.shift(3),
        "ReturnLag5": ret.shift(5),
        "SMAShort": sma_short,
        "SMALong": sma_long,
        "SMADiff": sma_diff,
        "Volatility20": ret.rolling(20).std(),
        "ROC10": price / price.shift(10) - 1,
        "RSI14": compute_rsi(price, window=14),
    })

    if include_regime_features:
        features["EfficiencyRatio"] = regime.efficiency_ratio(price, window=config.REGIME_WINDOW)
        features["ClassicalSignal"] = np.where(sma_diff > 0, 1.0, -1.0)

    return features


def build_target(price, horizon=None):
    """Target is the CUMULATIVE simple return over the next `horizon` days
    (default 1, i.e. just tomorrow's return): price.pct_change(horizon)
    gives the trailing `horizon`-day return ending today; shift(-horizon)
    moves that value back so it lines up with day t's row instead, meaning
    row t's features (known at end of day t) line up with the return
    realized over day t -> day t+horizon. horizon=1 reduces to the original
    "tomorrow's return" target exactly (same row-index-vs-value-timing
    convention as pairs_trading/ml_common.py's build_target() for
    horizon=1). Trading is still done DAILY regardless of horizon -- only
    what the model is trained to predict changes; the sign of that
    (possibly multi-day) prediction is still used as tomorrow's signal (see
    finish_strategy() below), so this is purely a smoother/less-noisy
    training target, not a change to the trading rule itself."""
    horizon = horizon if horizon is not None else config.TARGET_HORIZON
    return price.pct_change(horizon).shift(-horizon)


def build_sequence_target(price, horizon=None):
    """Sequence-model (LSTM) analog of build_target() above, preserving
    build_sequences()'s existing relative-indexing convention EXACTLY (that
    function has always been called with the raw daily-return series and
    grabs target_series.iloc[i + 1] for a window ending at position i): at
    horizon=1 this returns exactly price.pct_change() (identical to what
    build_sequences() was always passed before), so LSTM results reproduce
    exactly when config.TARGET_HORIZON=1. For horizon>1, this generalizes
    to the CUMULATIVE return over the next `horizon` days measured from
    each window's last included day, while keeping the value positioned at
    the same day-label build_sequences() already reads it from -- i.e.
    build_sequences(features, build_sequence_target(price, h), seq_len)'s
    y_seq[k] equals build_target(price, h)'s value at the corresponding
    window's last day, just addressed via the +1-shifted index
    build_sequences() has always used internally."""
    horizon = horizon if horizon is not None else config.TARGET_HORIZON
    return price.pct_change(horizon).shift(-(horizon - 1))


def prepare_data(full_price, formation_end=None, trading_start=None, trading_end=None,
                  short_window=None, long_window=None, include_regime_features=False):
    """Shared setup for all flat-feature models (RF, SVM, NN): build
    features/target over the full history, and split into formation (train)
    vs trading (predict). Unlike pairs_trading's version, there's no
    formation-only model to fit first (no OLS hedge ratio here) -- SMA
    windows are fixed hyperparameters, so features can be built directly
    over the whole series."""
    formation_end = formation_end or config.FORMATION_END
    trading_start = trading_start or config.TRADING_START
    trading_end = trading_end or config.TRADING_END

    full_return = full_price.pct_change()
    full_features = build_features(full_price, short_window, long_window, include_regime_features)
    full_target = build_target(full_price)

    formation_features = full_features.loc[:formation_end]
    formation_target = full_target.loc[:formation_end]
    trading_features = full_features.loc[trading_start:trading_end]

    # Drop rows where rolling windows / lags haven't filled yet (only affects
    # the very start of the formation period in practice).
    train_data = pd.concat([formation_features, formation_target.rename("target")], axis=1).dropna()

    return {
        "full_return": full_return,
        "train_X": train_data.drop(columns="target"),
        "train_y": train_data["target"],
        "trading_features": trading_features,
        "trading_start": trading_start, "trading_end": trading_end,
    }


def finish_strategy(predicted_return, full_return, trading_start, trading_end):
    """Common tail end for every ML strategy: turn a predicted-next-day-
    return series into a +1/-1 signal (sign of the prediction), then daily
    returns -- identical trading rule to the classical strategy (see
    signals.py), the ONLY difference between "classical" and "ML" in this
    whole package is this one substitution (SMA-crossover sign vs
    predicted-return sign)."""
    actual_return = full_return.loc[trading_start:trading_end]
    predicted_return = predicted_return.reindex(actual_return.index)

    signal = pd.Series(np.where(predicted_return > 0, 1.0, -1.0), index=predicted_return.index)
    signal = signal.where(~predicted_return.isna())  # leave genuinely missing predictions as NaN, not a fake +1

    daily_returns = signals.strategy_returns(actual_return, signal)

    return {
        "predicted_return": predicted_return,
        "actual_return": actual_return,  # exposed so callers can compute RMSE / compare to a naive baseline
        "signal": signal,
        "daily_returns": daily_returns,
    }


def finish_strategy_sized(predicted_return, full_return, trading_start, trading_end, scale):
    """===================== CONFIDENCE-WEIGHTED POSITION SIZING (experiment) =====================
    Alternative to finish_strategy()'s flat +1/-1 signal: instead of always
    betting a full unit long or short, the position size is the predicted
    return normalized by `scale` (the FORMATION period's own target
    std -- passed in by the caller, so this stays causal: no trading-period
    information leaks into the sizing), then clipped to [-1, 1] -- a
    prediction of "0.5 formation-std-worths" of return gives a half-sized
    position, a prediction of "2+ formation-std-worths" gives a full-sized
    position in either direction, same as the flat rule would in that case.
    signals.strategy_returns() (signal.shift(1) * simple_return) is
    completely generic to any signal magnitude, not just +/-1, so nothing
    else needs to change -- this is a drop-in replacement for
    finish_strategy() when config.POSITION_SIZING is True (see each
    ml_*.py's run_strategy()).
    ==========================================================================================="""
    actual_return = full_return.loc[trading_start:trading_end]
    predicted_return = predicted_return.reindex(actual_return.index)

    if scale is None or scale == 0 or np.isnan(scale):
        position = pd.Series(np.where(predicted_return > 0, 1.0, -1.0), index=predicted_return.index)
    else:
        position = (predicted_return / scale).clip(-1.0, 1.0)
    position = position.where(~predicted_return.isna())

    daily_returns = signals.strategy_returns(actual_return, position)

    return {
        "predicted_return": predicted_return,
        "actual_return": actual_return,
        "signal": position,
        "daily_returns": daily_returns,
    }


def finish_strategy_auto(predicted_return, full_return, trading_start, trading_end, train_y,
                          position_sizing=None):
    """Dispatches to finish_strategy() or finish_strategy_sized() depending
    on config.POSITION_SIZING (or an explicit override), computing `scale`
    as train_y's std (the formation target's own volatility -- causal, no
    trading-period data involved) automatically so every ml_*.py's
    run_strategy() only needs one call here instead of an if/else."""
    position_sizing = position_sizing if position_sizing is not None else config.POSITION_SIZING
    if position_sizing:
        scale = train_y.std()
        return finish_strategy_sized(predicted_return, full_return, trading_start, trading_end, scale)
    return finish_strategy(predicted_return, full_return, trading_start, trading_end)


def naive_persistence_forecast(full_return, trading_start, trading_end):
    """===================== NAIVE BASELINE (persistence forecast) =====================
    The simplest non-trivial forecast: tomorrow's return = today's return
    (i.e. betting that whatever just happened keeps happening -- literally
    the null hypothesis a trend-following strategy is implicitly relying on,
    made explicit as its own baseline). Any ML model that can't beat this on
    RMSE isn't adding real predictive signal over the crudest possible
    momentum rule -- worth reporting alongside every model's own RMSE (see
    prediction_rmse()). Note this is a DIFFERENT (and arguably fairer, less
    trivial) reference point than a flat "predict zero return" forecast
    would be, since returns are much less autocorrelated than price/spread
    LEVELS -- worth stating explicitly in the write-up.
    ======================================================================================"""
    predicted = full_return.shift(1).loc[trading_start:trading_end]
    actual = full_return.loc[trading_start:trading_end]
    return predicted, actual


def run_naive_baseline(full_price, formation_end=None, trading_start=None, trading_end=None,
                        include_regime_features=None):
    """The naive persistence baseline (see naive_persistence_forecast()),
    framed with the SAME (price, formation_end, trading_start, trading_end)
    signature every other strategy here uses, purely so it can be run
    through walk_forward.run_walk_forward() the same way and stay
    period-consistent with whatever config.WALK_FORWARD is set to. Not a
    tradeable strategy -- has no daily_returns, just predicted_return /
    actual_return -- used only to give every real model's own RMSE an
    honest, non-trivial reference point to beat. include_regime_features is
    accepted-but-ignored, same reason as classical_strategy.py's own
    unused-but-accepted param -- see main.py's maybe_walk_forward()."""
    trading_start = trading_start or config.TRADING_START
    trading_end = trading_end or config.TRADING_END
    full_return = full_price.pct_change()
    predicted, actual = naive_persistence_forecast(full_return, trading_start, trading_end)
    return {"predicted_return": predicted, "actual_return": actual}


def prediction_rmse(predicted_return, actual_return):
    """RMSE between any predicted-return series and the actual return,
    aligned on shared (non-NaN) dates. Used for both each ML model's own
    predictions and the naive persistence baseline above, so they're
    directly comparable."""
    aligned = pd.concat(
        [predicted_return.rename("pred"), actual_return.rename("actual")], axis=1
    ).dropna()
    if aligned.empty:
        return np.nan
    return float(np.sqrt(((aligned["pred"] - aligned["actual"]) ** 2).mean()))


def standardize_features(train_X, other_X_list):
    """===================== FEATURE STANDARDIZATION (flat models) =====================
    Fits a StandardScaler on the FORMATION period only (no look-ahead --
    the trading period never influences the mean/std used to rescale it),
    then applies that same fitted transform to every other DataFrame passed
    in (typically the trading-period features). Same rationale as
    pairs_trading/ml_common.py's version: irrelevant to Random Forest, but
    matters for SVM/Neural Net/LSTM given features like SMAShort/SMALong
    (raw price-level units, can be in the tens of thousands for BTC) sitting
    next to Return (typically +/-0.05) or SMADiff (a small ratio).
    ===================================================================================="""
    scaler = StandardScaler()
    train_scaled = pd.DataFrame(
        scaler.fit_transform(train_X), index=train_X.index, columns=train_X.columns
    )
    others_scaled = [
        pd.DataFrame(scaler.transform(X), index=X.index, columns=X.columns)
        for X in other_X_list
    ]
    return train_scaled, others_scaled, scaler


def standardize_sequences(X_seq_train, X_seq_other_list):
    """Same idea as standardize_features(), but for the 3D sequence arrays
    (n_samples, seq_len, n_features) used by the LSTM. The scaler is fit on
    the formation sequences only, flattened across samples and time steps,
    then applied to every array in X_seq_other_list."""
    n_features = X_seq_train.shape[2]
    scaler = StandardScaler()
    scaler.fit(X_seq_train.reshape(-1, n_features))

    def _transform(X_seq):
        shape = X_seq.shape
        flat_scaled = scaler.transform(X_seq.reshape(-1, n_features))
        return flat_scaled.reshape(shape)

    train_scaled = _transform(X_seq_train)
    others_scaled = [_transform(X) for X in X_seq_other_list]
    return train_scaled, others_scaled, scaler


def train_validation_split_flat(X, y, validation_fraction=0.2):
    """Chronological (NOT shuffled -- this is time series) split of the
    formation period into a training slice and a validation slice, used for
    early stopping in ml_neural_net.py. The validation slice is the LAST
    validation_fraction of the formation period."""
    n = len(X)
    split_idx = int(n * (1 - validation_fraction))
    return X.iloc[:split_idx], y.iloc[:split_idx], X.iloc[split_idx:], y.iloc[split_idx:]


def train_validation_split_sequences(X_seq, y_seq, validation_fraction=0.2):
    """Same idea as train_validation_split_flat(), for the 3D sequence
    arrays used by ml_lstm.py."""
    n = len(X_seq)
    split_idx = int(n * (1 - validation_fraction))
    return X_seq[:split_idx], y_seq[:split_idx], X_seq[split_idx:], y_seq[split_idx:]


def check_convergence(loss_history, model_name, window=10, tolerance=0.01):
    """Shared by ml_neural_net.py and ml_lstm.py: prints a warning if the
    training loss is still improving by more than `tolerance` (relative)
    over the last `window` epochs -- a quick, honest diagnostic rather than
    silently trusting a fixed epoch count."""
    if len(loss_history) < window + 1:
        return
    recent_start, recent_end = loss_history[-window - 1], loss_history[-1]
    if recent_start == 0:
        return
    relative_change = abs(recent_start - recent_end) / abs(recent_start)
    if relative_change > tolerance:
        print(f"  WARNING: {model_name} training loss changed by {relative_change:.1%} "
              f"over the last {window} epochs -- likely hasn't converged. Consider raising EPOCHS.")
    else:
        print(f"  {model_name} training loss changed by only {relative_change:.1%} "
              f"over the last {window} epochs -- looks converged.")


def build_sequences(features_df, target_series, seq_len):
    """For sequence models (LSTM): turn a flat (n_rows, n_features) frame
    into overlapping windows of shape (n_samples, seq_len, n_features).

    features_df and target_series should share the same date index (pass
    the FULL history, formation+trading concatenated -- target_series should
    be the RAW return series, NOT the pre-shifted 'target' used by the flat
    models). Window [i-seq_len+1, ..., i] (using only information known
    through day i) predicts target_series at i+1 (tomorrow) -- same
    off-by-one-avoidance convention as pairs_trading/ml_common.py's
    build_sequences().

    Returns (X_seq, y_seq, target_dates) as numpy arrays / a DatetimeIndex,
    where target_dates[k] is the actual calendar date the k-th prediction
    corresponds to -- safe to reindex predictions back onto later."""
    common_idx = features_df.dropna().index.intersection(target_series.dropna().index)
    features_df = features_df.loc[common_idx]
    target_series = target_series.loc[common_idx]

    values = features_df.values
    dates = features_df.index

    X_seq, y_seq, target_dates = [], [], []
    for i in range(seq_len - 1, len(dates) - 1):  # need i+1 to exist
        window = values[i - seq_len + 1: i + 1]      # seq_len rows ending at position i
        X_seq.append(window)
        y_seq.append(target_series.iloc[i + 1])       # tomorrow's actual return
        target_dates.append(dates[i + 1])             # tomorrow's date

    return np.array(X_seq), np.array(y_seq), pd.DatetimeIndex(target_dates)
