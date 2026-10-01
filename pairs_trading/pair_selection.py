"""
Within-sector cointegration screening (Engle-Granger, via
statsmodels.tsa.stattools.coint) -- Peng Liu Ch.8 methodology, restricted to
same-sector candidate pairs only.

See DISSERTATION_PLAN.md for the full methodological justification: testing
every possible S&P 500 pair (~124,750 combinations) at a 5% significance
threshold would be expected to produce thousands of "significant" pairs by
chance alone (a classic multiple-testing / false-discovery problem), on top
of giving no economic rationale for why any given pair should mean-revert.
Restricting to within-sector combinations shrinks the test space and gives
each surviving pair a plausible common-fundamentals story.
"""

import itertools
import pandas as pd
from statsmodels.tsa.stattools import coint

import config
import data_utils


def screen_pairs(formation_prices, permnos, pvalue_threshold=None, min_overlap_days=500, sector_label=None):
    """Test every within-sector pair on the FORMATION period only. Returns a
    DataFrame sorted by p-value (best/most cointegrated first).

    sector_label is purely cosmetic (used in the printed progress messages
    below) -- pass the sector name being screened when calling this in a
    loop over config.SECTORS so the console output is unambiguous about
    which sector's candidates are being tested. Defaults to "sector" if not
    given (e.g. when called standalone via this file's __main__ block).

    min_overlap_days defaults to ~2 years of trading days. This matters more
    than it sounds: a company that IPO'd partway through the formation
    period (e.g. Robinhood/HOOD, July 2021) only has a few months of
    overlapping history with an older stock. A cointegration test on that
    short a window is statistically underpowered and prone to spurious
    "significant" results -- confirmed empirically here: several HOOD pairs
    showed up in the top results before this filter was added, purely
    because of the short overlap, not a genuine long-run relationship.
    """
    pvalue_threshold = pvalue_threshold if pvalue_threshold is not None else config.COINT_PVALUE_THRESHOLD
    sector_label = sector_label or "sector"

    candidates = [p for p in permnos if p in formation_prices.columns]
    dropped = set(permnos) - set(candidates)
    if dropped:
        print(f"Note: {len(dropped)} PERMNOs in this sector have no price data in the formation period, skipping: {dropped}")

    pairs = list(itertools.combinations(candidates, 2))
    n_tests = len(pairs)
    expected_false_positives = n_tests * pvalue_threshold
    print(f"Screening {n_tests} candidate pairs within sector '{sector_label}' "
          f"at p < {pvalue_threshold} "
          f"(naive expectation if NONE were truly cointegrated: ~{expected_false_positives:.1f} "
          f"would still look 'significant' by chance -- report this alongside results).")

    results = []
    skipped_short_overlap = 0
    for p1, p2 in pairs:
        s1 = formation_prices[p1].dropna()
        s2 = formation_prices[p2].dropna()
        common_idx = s1.index.intersection(s2.index)
        if len(common_idx) < min_overlap_days:
            skipped_short_overlap += 1
            continue
        s1, s2 = s1.loc[common_idx], s2.loc[common_idx]

        # ===================== COINTEGRATION TEST =====================
        # Engle-Granger test (statsmodels' coint() runs the full two-step
        # procedure internally). Null hypothesis: the two price series are
        # NOT cointegrated. A low p-value lets us reject that, i.e. gives
        # evidence of a stable, mean-reverting relationship worth trading.
        # ================================================================
        try:
            score, pvalue, _ = coint(s1.values, s2.values)
        except Exception:
            continue
        results.append({
            "permno1": p1, "permno2": p2, "coint_score": score, "pvalue": pvalue,
            "overlap_days": len(common_idx),
        })

    print(f"Skipped {skipped_short_overlap} pairs with < {min_overlap_days} days of overlapping "
          f"formation-period history (e.g. recent IPOs) -- avoids spurious short-window results.")

    result_df = pd.DataFrame(results).sort_values("pvalue").reset_index(drop=True)
    significant = result_df[result_df["pvalue"] < pvalue_threshold]
    print(f"{len(significant)} of {len(result_df)} testable pairs are significant at p < {pvalue_threshold} "
          f"(vs. ~{expected_false_positives:.1f} expected by chance alone -- see note above).")

    # ===================== MULTIPLE-TESTING CHECK (Benjamini-Hochberg) =====================
    # The raw p<0.05 count above does not control the false discovery rate across ~2,700
    # simultaneous tests. Benjamini-Hochberg (1995) answers a stricter question: sorting all
    # p-values ascending, find the largest rank k such that p_(k) <= (k/m)*q -- every pair at
    # or above that rank is a "discovery" at FDR level q. Reporting both numbers side by side
    # (raw-significant vs. FDR-surviving) is standard practice whenever many hypotheses are
    # tested at once, and is worth stating explicitly in the methodology/limitations section:
    # a low or zero FDR-survival count does not mean the screened pairs are economically
    # useless (this project follows Gatev, Goetzmann & Rouwenhorst (2006)'s ranking-based
    # pair selection, not a formal family-wise hypothesis test), but it does mean the p-value
    # ranking should be read as an economically-motivated ordering device rather than a
    # family-wise-error-controlled significance claim.
    # ========================================================================================
    m = len(result_df)
    ranks = pd.Series(range(1, m + 1), index=result_df.index)
    for q in (0.05, 0.10):
        bh_threshold = (ranks / m) * q
        passes = result_df["pvalue"].values <= bh_threshold.values
        k_star = ranks[passes].max() if passes.any() else 0
        print(f"Benjamini-Hochberg FDR check at q={q}: {k_star} of {m} pairs survive "
              f"(i.e. top {k_star} pairs by p-value are 'discoveries' at this FDR level).")

    return result_df


def top_pairs_with_tickers(result_df, n=None):
    """Attach readable tickers/names to the top N pairs for reporting."""
    n = n if n is not None else config.PAIRS_PER_SECTOR
    sector_map = data_utils.load_sector_map()
    top = result_df.head(n).copy()
    top["ticker1"] = top["permno1"].apply(lambda p: data_utils.ticker_for(p, sector_map))
    top["ticker2"] = top["permno2"].apply(lambda p: data_utils.ticker_for(p, sector_map))
    return top


if __name__ == "__main__":
    prices = data_utils.load_wide_prices()
    permnos = data_utils.get_sector_permnos()
    formation, trading = data_utils.split_formation_trading(prices)

    results = screen_pairs(formation, permnos)
    top = top_pairs_with_tickers(results)
    print("\nTop pairs:")
    print(top[["ticker1", "ticker2", "coint_score", "pvalue"]])
