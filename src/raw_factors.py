"""
Robustness extension 4: the main tests on raw, non-winsorised factors.
Specification fixed in robustness_spec.py before this was run.

Momentum, reversal and volatility are used without the monthly
1st/99th percentile clip (size was never winsorised). The four-factor
multivariate Fama-MacBeth regression, the market-adjusted alphas of its
slopes, and the decile long-short spreads are re-estimated; nothing else
changes.

Decile sorts depend only on ranks, so clipping can change them only by
creating ties at the cut-offs. The regression uses factor magnitudes, so
it can change materially: an unclipped extreme value (SanDisk's 5,000%
momentum) dominates the cross-sectional standard deviation used to
z-score the factor and pulls the fitted slope towards that one stock.
"""

import pandas as pd

from decile_backtest import PRICES_PATH, monthly_returns
from extensions import decile_backtest
from fama_macbeth import cross_sectional_regressions, time_series_test
from risk_adjustment import capm, market_excess
from robustness_spec import RAW_FACTOR_FILES
from validation_costs import spread_stats

OUT_DIR = "output"
FACTORS = list(RAW_FACTOR_FILES)

if __name__ == "__main__":
    prices = pd.read_csv(PRICES_PATH, index_col=0, parse_dates=True)
    returns = monthly_returns(prices)
    factors = {k: pd.read_csv(p, index_col=0, parse_dates=True) for k, p in RAW_FACTOR_FILES.items()}

    ls = pd.DataFrame({k: decile_backtest(factors[k], returns)[0] for k in FACTORS})
    slopes, r2, n_obs = cross_sectional_regressions(returns, factors)
    slopes.to_csv(f"{OUT_DIR}/raw_fm_slopes_multivariate.csv")
    ls.to_csv(f"{OUT_DIR}/raw_long_short_returns.csv")

    main_ls = pd.read_csv(f"{OUT_DIR}/long_short_returns.csv", index_col=0, parse_dates=True)
    main_slopes = pd.read_csv(f"{OUT_DIR}/fm_slopes_multivariate.csv", index_col=0, parse_dates=True)

    def block(ls_, slopes_):
        st, fm = spread_stats(ls_), time_series_test(slopes_[FACTORS])
        al = capm(slopes_[FACTORS], market_excess(slopes_.index)[0])
        return pd.DataFrame({"ls_ann_ret_%": st["ann_ret_%"], "ls_sharpe": st["sharpe"], "ls_nw_t": st["nw_t"],
                             "fm_mean_%": fm["mean_%"], "fm_nw_t": fm["nw_t"],
                             "fm_mktadj_alpha_%": al["alpha_ann_%"], "fm_mktadj_t": al["alpha_nw_t"]})

    comparison = pd.concat({"winsorised": block(main_ls, main_slopes), "raw": block(ls, slopes)},
                           names=["treatment", "factor"]).astype(float)
    comparison.to_csv(f"{OUT_DIR}/raw_factor_comparison.csv")

    pd.set_option("display.width", 220)
    print(f"Raw-factor FM: {len(slopes)} months, avg {n_obs.mean():.0f} stocks, avg R^2 {r2.mean():.3f}")
    print(comparison.round(3).unstack("treatment").T)
    print("\nCorrelation of monthly slopes, winsorised vs raw:")
    print(main_slopes[FACTORS].corrwith(slopes[FACTORS]).round(3))
    print("\nCorrelation of long-short returns, winsorised vs raw:")
    print(main_ls[FACTORS].corrwith(ls[FACTORS]).round(4))
