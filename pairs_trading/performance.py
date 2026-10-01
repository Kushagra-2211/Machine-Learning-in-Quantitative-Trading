"""
Shared performance metrics and comparison table for the three strategies.
"""

import numpy as np
import pandas as pd

import config

# Model-capacity grouping used by family_comparison() below, motivated by
# Section 2.4's structural hypothesis: Random Forest, Gradient Boosting and
# the LSTM have more capacity to learn nonlinear feature interactions than a
# linear-kernel SVM or a small feed-forward Neural Net, so if the theory is
# right, any ML-vs-Classical edge (and its regime-dependence) should be
# stronger and more concentrated in the first group than the second.
MODEL_FAMILY = {
    "ML: Random Forest": "High-capacity (tree/sequence)",
    "ML: Gradient Boosting": "High-capacity (tree/sequence)",
    "ML: LSTM": "High-capacity (tree/sequence)",
    "ML: SVM": "Simple (linear/shallow)",
    "ML: Neural Net": "Simple (linear/shallow)",
}


def cumulative_return(daily_returns):
    return (1 + daily_returns).cumprod()


def terminal_return(daily_returns):
    return cumulative_return(daily_returns).iloc[-1] - 1


def annualized_return(daily_returns, periods_per_year=None):
    periods_per_year = periods_per_year or config.TRADING_DAYS_PER_YEAR
    n = len(daily_returns)
    if n == 0:
        return np.nan
    total_growth = cumulative_return(daily_returns).iloc[-1]
    years = n / periods_per_year
    if years == 0 or total_growth <= 0:
        return np.nan
    return total_growth ** (1 / years) - 1


def annualized_volatility(daily_returns, periods_per_year=None):
    periods_per_year = periods_per_year or config.TRADING_DAYS_PER_YEAR
    return daily_returns.std() * np.sqrt(periods_per_year)


def sharpe_ratio(daily_returns, risk_free_rate=None, periods_per_year=None):
    risk_free_rate = risk_free_rate if risk_free_rate is not None else config.RISK_FREE_RATE
    periods_per_year = periods_per_year or config.TRADING_DAYS_PER_YEAR
    excess = daily_returns - risk_free_rate / periods_per_year
    vol = excess.std()
    if vol == 0 or np.isnan(vol):
        return np.nan
    return (excess.mean() / vol) * np.sqrt(periods_per_year)


def max_drawdown(daily_returns):
    cum = cumulative_return(daily_returns)
    running_max = cum.cummax()
    drawdown = (cum - running_max) / running_max
    return drawdown.min()


def summarize(strategy_returns_dict):
    """strategy_returns_dict: {"Classical": daily_returns_series,
    "ML": daily_returns_series, "Buy & Hold": daily_returns_series}"""
    rows = []
    for name, daily_returns in strategy_returns_dict.items():
        rows.append({
            "Strategy": name,
            "Terminal Return": terminal_return(daily_returns),
            "Annualized Return": annualized_return(daily_returns),
            "Annualized Volatility": annualized_volatility(daily_returns),
            "Sharpe Ratio": sharpe_ratio(daily_returns),
            "Max Drawdown": max_drawdown(daily_returns),
        })
    return pd.DataFrame(rows).set_index("Strategy")


# ===================== DISPERSION / SIGNIFICANCE (across pairs) =====================
# Added after a 5-pair backtest where one pair (GPN-FIS) had a bad trading
# period for EVERY strategy, including Classical and Buy & Hold -- with so
# few pairs, a single equal-weighted "portfolio" Sharpe can be dominated by
# one unlucky (or lucky) pair rather than reflecting a genuine, repeatable
# model difference. These functions look at the SPREAD of per-pair results,
# not just the pooled mean, and put a rough uncertainty band and a
# significance check around any "model A beat model B" claim.
# =======================================================================================
def per_pair_metric_table(all_pair_results, metric_fn=None):
    """all_pair_results: {pair_label: {strategy_name: daily_returns_series}}
    (this is exactly main.py's all_pair_results). Returns a DataFrame, rows =
    pairs, columns = strategies, values = the given metric (default Sharpe
    ratio) computed independently on EACH pair's own daily returns."""
    metric_fn = metric_fn or sharpe_ratio
    rows = {}
    for pair_label, strategy_returns in all_pair_results.items():
        rows[pair_label] = {name: metric_fn(returns) for name, returns in strategy_returns.items()}
    return pd.DataFrame(rows).T


def dispersion_summary(all_pair_results, metric_fn=None, metric_name="Sharpe Ratio"):
    """Mean/std/min/max of a per-pair metric across all pairs, per strategy
    -- report alongside the pooled portfolio number in summarize(), not
    instead of it. A strategy with a similar mean but much larger std than
    another is a materially different (less consistent) result even if the
    pooled Sharpe looks similar."""
    metric_fn = metric_fn or sharpe_ratio
    table = per_pair_metric_table(all_pair_results, metric_fn)
    return pd.DataFrame({
        f"Mean {metric_name}": table.mean(),
        f"Std {metric_name}": table.std(),
        f"Min {metric_name}": table.min(),
        f"Max {metric_name}": table.max(),
        "N Pairs": table.count(),
    })


def bootstrap_sharpe_ci(daily_returns, n_bootstrap=2000, ci=0.95, risk_free_rate=None,
                         periods_per_year=None, random_state=None):
    """i.i.d. bootstrap confidence interval on the Sharpe ratio: resample
    daily returns WITH replacement (same length as the original series)
    n_bootstrap times, recompute Sharpe each time, and take the (1-ci)/2 and
    (1+ci)/2 percentiles as the interval.

    Caveat: this ignores serial correlation in daily strategy returns
    (position-holding creates autocorrelation) -- kept here for reference
    and comparison, but block_bootstrap_sharpe_ci() below is the version
    actually used for reported confidence intervals, since it preserves
    that serial correlation instead of assuming it away."""
    random_state = random_state if random_state is not None else config.RANDOM_SEED
    rng = np.random.default_rng(random_state)
    values = daily_returns.dropna().values
    n = len(values)
    point_estimate = sharpe_ratio(daily_returns, risk_free_rate, periods_per_year)
    if n == 0:
        return point_estimate, np.nan, np.nan

    boot_sharpes = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sample = rng.choice(values, size=n, replace=True)
        boot_sharpes[i] = sharpe_ratio(pd.Series(sample), risk_free_rate, periods_per_year)

    lower_pct, upper_pct = (1 - ci) / 2 * 100, (1 + ci) / 2 * 100
    lower, upper = np.nanpercentile(boot_sharpes, [lower_pct, upper_pct])
    return point_estimate, lower, upper


def block_bootstrap_sharpe_ci(daily_returns, n_bootstrap=2000, ci=0.95, risk_free_rate=None,
                               periods_per_year=None, block_size=None, random_state=None):
    """Moving-block bootstrap confidence interval on the Sharpe ratio: resample
    CONTIGUOUS blocks of `block_size` consecutive days (with replacement) and
    concatenate them until the resample reaches the original series' length,
    instead of resampling individual days independently as
    bootstrap_sharpe_ci() does. This preserves the day-to-day serial
    correlation that position-holding strategies -- both the pairs and
    momentum strands -- introduce into daily returns, which i.i.d. resampling
    destroys. block_size defaults to config.BLOCK_SIZE (20 periods) if not
    given explicitly. This is the version used for every reported confidence
    interval in this dissertation (see Methodology Section 4.6)."""
    random_state = random_state if random_state is not None else config.RANDOM_SEED
    block_size = block_size or getattr(config, "BLOCK_SIZE", 20)
    rng = np.random.default_rng(random_state)
    values = daily_returns.dropna().values
    n = len(values)
    point_estimate = sharpe_ratio(daily_returns, risk_free_rate, periods_per_year)
    if n == 0:
        return point_estimate, np.nan, np.nan

    block_size = max(1, min(block_size, n))
    n_blocks_needed = int(np.ceil(n / block_size))
    max_start = n - block_size  # last valid block start index

    boot_sharpes = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        if max_start <= 0:
            # Series shorter than one block: falls back to i.i.d. resampling
            # of individual days (nothing else is possible at this length).
            sample = rng.choice(values, size=n, replace=True)
        else:
            starts = rng.integers(0, max_start + 1, size=n_blocks_needed)
            blocks = [values[s:s + block_size] for s in starts]
            sample = np.concatenate(blocks)[:n]
        boot_sharpes[i] = sharpe_ratio(pd.Series(sample), risk_free_rate, periods_per_year)

    lower_pct, upper_pct = (1 - ci) / 2 * 100, (1 + ci) / 2 * 100
    lower, upper = np.nanpercentile(boot_sharpes, [lower_pct, upper_pct])
    return point_estimate, lower, upper


def benjamini_hochberg(pvalues, q=0.05):
    """Benjamini-Hochberg false-discovery-rate correction across a FAMILY of
    p-values tested together -- e.g. the block of ML-vs-Classical Wilcoxon
    tests produced by one call to report_regime_stats() in main.py (up to
    five models compared against Classical at once). Returns a boolean array
    in the same order as the input, True where that hypothesis remains
    significant after correcting for testing several models (and, across a
    full run, several regimes and universe sizes) simultaneously -- see
    Methodology Section 4.6 / the multiple-comparisons caveat in Discussion.
    NaN p-values (e.g. Classical's own row, which has no comparison) are
    left as False and excluded from the correction."""
    pvalues = np.asarray(pvalues, dtype=float)
    n = len(pvalues)
    reject = np.zeros(n, dtype=bool)
    valid = ~np.isnan(pvalues)
    m = valid.sum()
    if m == 0:
        return reject
    order = np.argsort(np.where(valid, pvalues, np.inf))
    sorted_p = pvalues[order]
    sorted_valid = valid[order]
    ranks = np.arange(1, n + 1)
    thresholds = (ranks / m) * q
    passing = np.where(sorted_valid & (sorted_p <= thresholds))[0]
    if len(passing) > 0:
        k = passing.max()
        reject[order[:k + 1]] = True
    return reject & valid


def family_comparison(all_pair_results, family_map=None, classical_name="Classical"):
    """Model-family breakdown motivated by Section 2.4's capacity argument:
    pools the per-pair Sharpe-ratio differences vs Classical across the ML
    models in each capacity family (see MODEL_FAMILY), then tests via a
    two-sided Mann-Whitney U test whether the two families' pooled
    differences differ from one another. Pooling models within a family
    multiplies the effective sample size for this test well beyond what any
    single model's per-pair Wilcoxon test has to work with (see the low-power
    caveat on paired_significance_test()).

    Call this on a regime-filtered all_pair_results (see
    filter_pair_results_by_regime()) to test the sharper, regime-conditional
    version of the same question: does the High-capacity family's edge
    concentrate specifically in the regime the structural hypothesis
    predicts it should?

    Returns a DataFrame (one row per family: mean/median pooled Sharpe diff,
    N pooled observations) with the cross-family Mann-Whitney p-value
    attached as result.attrs['family_comparison_p_value']."""
    from scipy.stats import mannwhitneyu
    family_map = family_map or MODEL_FAMILY
    table = per_pair_metric_table(all_pair_results)

    diffs_by_family = {}
    for strat_name, family in family_map.items():
        if strat_name not in table.columns or classical_name not in table.columns:
            continue
        diffs = (table[strat_name] - table[classical_name]).dropna()
        diffs_by_family.setdefault(family, []).extend(diffs.tolist())

    rows = []
    for family, diffs in diffs_by_family.items():
        arr = np.array(diffs, dtype=float)
        rows.append({
            "Model Family": family,
            "Mean Sharpe Diff vs Classical": arr.mean() if len(arr) else np.nan,
            "Median Sharpe Diff vs Classical": np.median(arr) if len(arr) else np.nan,
            "N Pooled Obs (pairs x models in family)": len(arr),
        })
    result = pd.DataFrame(rows).set_index("Model Family") if rows else pd.DataFrame(
        columns=["Mean Sharpe Diff vs Classical", "Median Sharpe Diff vs Classical",
                 "N Pooled Obs (pairs x models in family)"])

    families = list(diffs_by_family.keys())
    p_value = np.nan
    if len(families) == 2 and all(len(diffs_by_family[f]) >= 2 for f in families):
        try:
            _, p_value = mannwhitneyu(diffs_by_family[families[0]], diffs_by_family[families[1]],
                                       alternative="two-sided")
        except ValueError:
            p_value = np.nan
    result.attrs["family_comparison_p_value"] = p_value
    return result


# ===================== ANNUAL BREAKDOWN (regime dependence) =====================
# Added after noticing the portfolio cumulative-return chart shows spread-based
# strategies (Classical, SVM) tracking or beating Buy & Hold through the choppy
# 2022-2023 period (Fed hiking cycle + March 2023 regional-bank stress -- squarely
# within this project's Financials universe), then falling behind once a sustained
# rally set in from mid-2024. That is the textbook signature of pairs trading's
# known regime-dependence (Gatev, Goetzmann & Rouwenhorst, 2006; Do & Faff, 2010,
# 2012; Krauss, 2017 survey): mean-reversion strategies profit when a spread wanders
# and comes back, and structurally cannot profit when it trends persistently instead
# -- true of the trading rule itself regardless of whether the spread is predicted
# by OLS or by an ML model sitting on top of it. Breaking results down by calendar
# year turns that visual impression into an actual number per strategy per year.
# ===================================================================================
def annual_breakdown(strategy_returns_dict, risk_free_rate=None, periods_per_year=None):
    """strategy_returns_dict: {"Classical": daily_returns_series, ...} (works on
    either a single pair's daily_returns or the equal-weighted portfolio's --
    whichever series you pass in gets grouped by calendar year). Returns a
    DataFrame indexed by (Year, Strategy) with Terminal Return and Sharpe Ratio
    computed independently within each calendar year, so a strategy's edge (or
    lack of one) can be read off year by year rather than only as one pooled
    number across the whole 2022-2025 window."""
    rows = []
    for name, daily_returns in strategy_returns_dict.items():
        for year, year_returns in daily_returns.groupby(daily_returns.index.year):
            rows.append({
                "Year": year,
                "Strategy": name,
                "Terminal Return": terminal_return(year_returns),
                "Sharpe Ratio": sharpe_ratio(year_returns, risk_free_rate, periods_per_year),
                "N Days": len(year_returns),
            })
    return pd.DataFrame(rows).set_index(["Year", "Strategy"]).sort_index()


# ===================== REGIME-CONDITIONAL FILTERING =====================
# Pairs with regime.classify_regime() (see that module's docstring for the
# full rationale): once every trading day has a "Trending"/"Range-bound"
# label, restricting any daily_returns Series to one regime is just a
# reindex -- every function above (dispersion_summary, bootstrap_sharpe_ci,
# paired_significance_test) then works completely unchanged on the filtered
# series. This is deliberately the ONLY new function needed to run the whole
# existing stats suite conditioned on regime instead of unconditionally.
# ===========================================================================
def filter_returns_by_regime(daily_returns, regime_labels, regime_value):
    """Restrict daily_returns to the days classified as `regime_value`
    ("Trending" or "Range-bound") in regime_labels. Days present in
    daily_returns but absent from regime_labels (e.g. the warm-up window at
    the very start of the trading period, before enough history exists to
    compute the Efficiency Ratio) are dropped, same as any other date not in
    the target regime."""
    matching_dates = regime_labels[regime_labels == regime_value].index
    return daily_returns.reindex(daily_returns.index.intersection(matching_dates))


def filter_pair_results_by_regime(all_pair_results, regime_labels, regime_value):
    """all_pair_results: {pair_label: {strategy_name: daily_returns}} --
    returns the same nested structure, with every daily_returns Series
    restricted to the given regime via filter_returns_by_regime(). The
    result can be passed straight into per_pair_metric_table(),
    dispersion_summary(), or paired_significance_test() exactly as if it
    were a normal all_pair_results dict."""
    return {
        pair_label: {
            name: filter_returns_by_regime(returns, regime_labels, regime_value)
            for name, returns in strategy_returns.items()
        }
        for pair_label, strategy_returns in all_pair_results.items()
    }


# ===================== REGIME-INTERACTION BOOTSTRAP TESTS (RQ1/RQ3) =====================
# Everything above tests each regime SEPARATELY (report_regime_stats() in
# main.py runs the whole dispersion/bootstrap/Wilcoxon suite once on
# Trending days and once on Range-bound days) -- that shows each strategy's
# Sharpe ratio and its significance vs Classical WITHIN a regime, but never
# directly tests whether a strategy's OWN Sharpe differs BETWEEN regimes
# (RQ1), or whether an ML model's edge over Classical itself changes size
# between regimes (RQ3) -- both of those are read off two separate numbers
# by eye today. These two functions test each directly with one number and
# one p-value, using the SAME moving-block bootstrap (block_size, default
# config.BLOCK_SIZE) as block_bootstrap_sharpe_ci() above, so the whole
# dissertation stays internally consistent on one resampling method rather
# than switching to a regression/HAC alternative for just these two tests.
# ==========================================================================================
def bootstrap_two_sided_p_value(boot_values, null_value=0.0):
    """Two-sided bootstrap percentile p-value for a test statistic against
    `null_value` (0.0 for both tests below): the probability, under the
    bootstrap distribution, of a draw at least as extreme in EITHER
    direction as the null. p = 2 * min(P(boot >= null), P(boot <= null)),
    capped at 1 -- the standard nonparametric bootstrap analogue of a
    two-sided test when there's no reason to assume the statistic's
    sampling distribution is symmetric or normal."""
    boot_values = np.asarray(boot_values, dtype=float)
    boot_values = boot_values[~np.isnan(boot_values)]
    if len(boot_values) == 0:
        return np.nan
    p_ge = np.mean(boot_values >= null_value)
    p_le = np.mean(boot_values <= null_value)
    return min(1.0, 2 * min(p_ge, p_le))


def _joint_block_bootstrap_regime_stat(series_dict, regime_labels, statistic_fn,
                                        n_bootstrap=2000, block_size=None, random_state=None):
    """Shared resampling engine for both regime-interaction tests below.
    series_dict: {name: daily_returns_series}, all pre-aligned to the SAME
    index as regime_labels (the caller drops the handful of NaN-regime
    warm-up days first, same convention as filter_returns_by_regime()).
    Draws contiguous blocks of `block_size` days (default config.BLOCK_SIZE)
    -- the SAME block start positions applied to every series in
    series_dict AND to regime_labels simultaneously, so (a) same-day
    pairing across multiple series (needed for the joint ML-vs-Classical
    test) and (b) each resampled day's OWN actual regime label are both
    preserved intact in every replicate; the regime label is carried along
    with its day, never reshuffled independently of the return it belongs
    to. Calls statistic_fn(resampled_series_dict, resampled_labels) once
    per replicate and returns an array of length n_bootstrap (NaN where a
    replicate happens to contain no days of one regime, so the statistic
    can't be computed for that draw -- excluded by bootstrap_two_sided_p_value
    and np.nanpercentile rather than crashing the whole run)."""
    random_state = random_state if random_state is not None else config.RANDOM_SEED
    block_size = block_size or getattr(config, "BLOCK_SIZE", 20)
    rng = np.random.default_rng(random_state)

    common_index = regime_labels.index
    for s in series_dict.values():
        common_index = common_index.intersection(s.index)
    common_index = common_index.sort_values()

    labels_arr = regime_labels.reindex(common_index).values
    values_arr = {name: s.reindex(common_index).values for name, s in series_dict.items()}
    n = len(common_index)
    if n == 0:
        return np.full(n_bootstrap, np.nan)

    block_size = max(1, min(block_size, n))
    n_blocks_needed = int(np.ceil(n / block_size))
    max_start = n - block_size

    results = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        if max_start <= 0:
            idx = rng.integers(0, n, size=n)
        else:
            starts = rng.integers(0, max_start + 1, size=n_blocks_needed)
            idx = np.concatenate([np.arange(s, s + block_size) for s in starts])[:n]
        resampled_labels = pd.Series(labels_arr[idx])
        try:
            results[i] = statistic_fn(
                {name: pd.Series(vals[idx]) for name, vals in values_arr.items()},
                resampled_labels,
            )
        except (ValueError, ZeroDivisionError):
            results[i] = np.nan
    return results


def trend_range_sharpe_delta(daily_returns, regime_labels, risk_free_rate=None, periods_per_year=None):
    """Point estimate for the RQ1/Point-6 test: one strategy's own Sharpe
    ratio in Trending days minus its Sharpe ratio in Range-bound days,
    computed from the SAME daily_returns series filtered by regime_labels
    (see filter_returns_by_regime())."""
    trending = filter_returns_by_regime(daily_returns, regime_labels, "Trending")
    range_bound = filter_returns_by_regime(daily_returns, regime_labels, "Range-bound")
    return (sharpe_ratio(trending, risk_free_rate, periods_per_year)
            - sharpe_ratio(range_bound, risk_free_rate, periods_per_year))


def trend_range_delta_bootstrap_test(daily_returns, regime_labels, n_bootstrap=2000, ci=0.95,
                                      risk_free_rate=None, periods_per_year=None, block_size=None,
                                      random_state=None):
    """RQ1 / Point 6: is a single strategy's OWN Sharpe ratio itself
    regime-dependent? Each bootstrap draw resamples contiguous blocks from
    the strategy's daily-return series (carrying each resampled day's
    ACTUAL regime label along with it), recomputes Sharpe_Trending and
    Sharpe_Range-bound from that same resample, and takes their difference
    -- so the within-regime autocorrelation structure and the true
    day-to-regime mapping are both preserved in every replicate, unlike
    naively comparing two independently-bootstrapped Sharpe CIs (which
    would ignore that the two regime subsets are complementary halves of
    the SAME series). Returns (point_estimate, ci_lower, ci_upper,
    p_value); p_value tests H0: delta == 0 via bootstrap_two_sided_p_value()."""
    common = regime_labels.index.intersection(daily_returns.dropna().index)
    labels = regime_labels.reindex(common)
    returns = daily_returns.reindex(common)
    point_estimate = trend_range_sharpe_delta(returns, labels, risk_free_rate, periods_per_year)

    def stat_fn(series_dict, resampled_labels):
        return trend_range_sharpe_delta(series_dict["returns"], resampled_labels,
                                         risk_free_rate, periods_per_year)

    boot = _joint_block_bootstrap_regime_stat({"returns": returns}, labels, stat_fn,
                                               n_bootstrap=n_bootstrap, block_size=block_size,
                                               random_state=random_state)
    lower_pct, upper_pct = (1 - ci) / 2 * 100, (1 + ci) / 2 * 100
    lower, upper = np.nanpercentile(boot, [lower_pct, upper_pct])
    p_value = bootstrap_two_sided_p_value(boot)
    return point_estimate, lower, upper, p_value


def trend_range_delta_table(strategy_returns_dict, regime_labels, strategies=None, n_bootstrap=2000,
                             ci=0.95, risk_free_rate=None, periods_per_year=None, block_size=None,
                             random_state=None, bh_q=0.05):
    """Builds the RQ1/Point-6 summary table: one row per strategy in
    `strategies` (default: every key in strategy_returns_dict), columns for
    the point estimate, bootstrap CI, p-value, and BH-corrected
    significance flag across the strategy family -- see
    trend_range_delta_bootstrap_test()."""
    strategies = strategies or list(strategy_returns_dict.keys())
    rows = []
    for name in strategies:
        point, lower, upper, p = trend_range_delta_bootstrap_test(
            strategy_returns_dict[name], regime_labels, n_bootstrap=n_bootstrap, ci=ci,
            risk_free_rate=risk_free_rate, periods_per_year=periods_per_year,
            block_size=block_size, random_state=random_state)
        rows.append({
            "Strategy": name,
            "Delta Sharpe (Trend - Range)": point,
            "CI Lower": lower,
            "CI Upper": upper,
            "p-value": p,
        })
    table = pd.DataFrame(rows).set_index("Strategy")
    table["BH Significant (q={:.2f})".format(bh_q)] = benjamini_hochberg(
        table["p-value"].astype(float).values, q=bh_q)
    return table


def ml_classical_regime_interaction(ml_returns, classical_returns, regime_labels,
                                     risk_free_rate=None, periods_per_year=None):
    """Point estimate for the RQ3/Point-5 test: Delta_regime =
    (Sharpe_ML - Sharpe_Classical)_Trending - (Sharpe_ML -
    Sharpe_Classical)_Range-bound -- i.e. does the SIZE of an ML model's
    edge (or deficit) over Classical itself change between regimes, tested
    as a single interaction term rather than compared by eye across two
    separately-reported per-regime Wilcoxon results."""
    trending_ml = filter_returns_by_regime(ml_returns, regime_labels, "Trending")
    trending_cl = filter_returns_by_regime(classical_returns, regime_labels, "Trending")
    range_ml = filter_returns_by_regime(ml_returns, regime_labels, "Range-bound")
    range_cl = filter_returns_by_regime(classical_returns, regime_labels, "Range-bound")
    trend_gap = (sharpe_ratio(trending_ml, risk_free_rate, periods_per_year)
                 - sharpe_ratio(trending_cl, risk_free_rate, periods_per_year))
    range_gap = (sharpe_ratio(range_ml, risk_free_rate, periods_per_year)
                 - sharpe_ratio(range_cl, risk_free_rate, periods_per_year))
    return trend_gap - range_gap


def ml_classical_regime_interaction_bootstrap_test(ml_returns, classical_returns, regime_labels,
                                                    n_bootstrap=2000, ci=0.95, risk_free_rate=None,
                                                    periods_per_year=None, block_size=None,
                                                    random_state=None):
    """RQ3 / Point 5: block-bootstrap test of the ML-vs-Classical
    INTERACTION with regime. Bootstraps the ML and Classical daily-return
    series JOINTLY -- the same block draws applied to both series and to
    regime_labels at once, so same-day pairing between the two strategies
    AND each day's true regime membership are both preserved in every
    replicate (this is what makes it a genuinely joint test rather than two
    separate single-series bootstraps combined after the fact). Returns
    (point_estimate, ci_lower, ci_upper, p_value)."""
    common = (regime_labels.index
              .intersection(ml_returns.dropna().index)
              .intersection(classical_returns.dropna().index))
    labels = regime_labels.reindex(common)
    ml = ml_returns.reindex(common)
    cl = classical_returns.reindex(common)
    point_estimate = ml_classical_regime_interaction(ml, cl, labels, risk_free_rate, periods_per_year)

    def stat_fn(series_dict, resampled_labels):
        return ml_classical_regime_interaction(series_dict["ml"], series_dict["classical"],
                                                resampled_labels, risk_free_rate, periods_per_year)

    boot = _joint_block_bootstrap_regime_stat({"ml": ml, "classical": cl}, labels, stat_fn,
                                               n_bootstrap=n_bootstrap, block_size=block_size,
                                               random_state=random_state)
    lower_pct, upper_pct = (1 - ci) / 2 * 100, (1 + ci) / 2 * 100
    lower, upper = np.nanpercentile(boot, [lower_pct, upper_pct])
    p_value = bootstrap_two_sided_p_value(boot)
    return point_estimate, lower, upper, p_value


def ml_classical_interaction_table(strategy_returns_dict, regime_labels, classical_name="Classical",
                                    ml_names=None, n_bootstrap=2000, ci=0.95, risk_free_rate=None,
                                    periods_per_year=None, block_size=None, random_state=None, bh_q=0.05):
    """Builds the RQ3/Point-5 summary table: one row per ML model in
    `ml_names` (default: every strategy except Classical and Buy & Hold),
    columns for the interaction point estimate, bootstrap CI, p-value, and
    BH-corrected significance flag across the ML-model family -- see
    ml_classical_regime_interaction_bootstrap_test()."""
    if ml_names is None:
        ml_names = [name for name in strategy_returns_dict
                    if name not in (classical_name, "Buy & Hold")]
    rows = []
    for name in ml_names:
        point, lower, upper, p = ml_classical_regime_interaction_bootstrap_test(
            strategy_returns_dict[name], strategy_returns_dict[classical_name], regime_labels,
            n_bootstrap=n_bootstrap, ci=ci, risk_free_rate=risk_free_rate,
            periods_per_year=periods_per_year, block_size=block_size, random_state=random_state)
        rows.append({
            "Model": name,
            "Delta_regime (Trend gap - Range gap)": point,
            "CI Lower": lower,
            "CI Upper": upper,
            "p-value": p,
        })
    table = pd.DataFrame(rows).set_index("Model")
    table["BH Significant (q={:.2f})".format(bh_q)] = benjamini_hochberg(
        table["p-value"].astype(float).values, q=bh_q)
    return table


def paired_significance_test(all_pair_results, strategy_a, strategy_b, metric_fn=None):
    """Wilcoxon signed-rank test on the per-pair metric difference between
    two strategies (nonparametric -- more appropriate than a paired t-test
    given how few pairs there are and no reason to assume the differences
    are normally distributed). Answers: "is strategy_a's edge over
    strategy_b consistent across pairs, or could it plausibly just be
    noise?" Returns (median_difference, p_value).

    Caveat worth stating alongside any result: with only a handful of pairs
    this test has low statistical power -- a non-significant p-value here
    means "not enough pairs to be sure", not "no difference exists"."""
    from scipy.stats import wilcoxon
    table = per_pair_metric_table(all_pair_results, metric_fn)
    diffs = (table[strategy_a] - table[strategy_b]).dropna()
    if len(diffs) < 2 or (diffs == 0).all():
        return diffs.median() if len(diffs) else np.nan, np.nan
    stat, p_value = wilcoxon(diffs)
    return diffs.median(), p_value
