"""
Stage 6: extensions.

(a) Sector-neutral factors. The universe has sectors with as few as 5
    stocks, too few for within-sector deciles. Instead each stock's factor
    is converted to a percentile rank *within its GICS sector* each month,
    and deciles are formed on that pooled within-sector percentile. Every
    decile then holds a similar sector mix, so the long-short spread is
    close to sector-neutral. The Fama-MacBeth counterpart adds sector
    dummies, so slopes use only within-sector variation.

(b) VIX regimes. Each month t is labelled by the VIX close at the end of
    month t-1 (known before month t's returns), split into full-sample
    terciles. Because the tercile cut-offs use the whole sample, this is a
    descriptive conditioning analysis, not a tradable timing rule.
    Differences between high- and low-VIX months are tested with a Welch
    t-test (unequal variances).

Sector labels are today's GICS classification, applied to the whole
history (a stated limitation: e.g. the 2018 GICS reshuffle that created
Communication Services is not reflected historically).
"""

import numpy as np
import pandas as pd
from scipy import stats

from decile_backtest import FACTOR_FILES, N_DECILES, PRICES_PATH, assign_deciles, long_short_weights
from fama_macbeth import cross_sectional_regressions, time_series_test
from validation_costs import COST_BPS, net_returns, spread_stats

UNIVERSE_PATH = "Project-1/data/sp500_top200.csv"
VIX_PATH = "Project-1/data/vix.csv"
OUT_DIR = "Project-1/output"
REGIMES = ["low", "mid", "high"]


def within_sector_percentile(factor, sectors):
    """Percentile rank of each stock's factor value among its own sector, per month."""
    by_sector = factor.T.groupby(sectors.reindex(factor.columns)).rank(pct=True)
    return by_sector.T[factor.columns]


def decile_backtest(factor, returns):
    """Long-short D10 - D1 returns and turnover, same rules as Stage 3."""
    deciles = assign_deciles(factor)
    top = returns.where(deciles == N_DECILES).mean(axis=1)
    bottom = returns.where(deciles == 1).mean(axis=1)
    ls = (top - bottom).dropna()
    weights = long_short_weights(deciles).loc[ls.index]
    turnover = weights.diff().abs().sum(axis=1).iloc[1:]
    return ls, turnover


def regime_table(series_by_name, regime):
    """Mean monthly return (%) per VIX tercile, plus high-vs-low Welch test."""
    rows = {}
    for name, s in series_by_name.items():
        s = s.dropna()
        r = regime.reindex(s.index)
        row = {f"{g}_mean_%": s[r == g].mean() * 100 for g in REGIMES}
        row.update({f"{g}_months": (r == g).sum() for g in REGIMES})
        t, p = stats.ttest_ind(s[r == "high"], s[r == "low"], equal_var=False)
        row.update({"high_minus_low_%": row["high_mean_%"] - row["low_mean_%"], "welch_t": t, "welch_p": p})
        rows[name] = row
    return pd.DataFrame(rows).T


if __name__ == "__main__":
    prices = pd.read_csv(PRICES_PATH, index_col=0, parse_dates=True)
    monthly_prices = prices.resample("ME").last()
    returns = monthly_prices.pct_change(fill_method=None)

    universe = pd.read_csv(UNIVERSE_PATH)
    sectors = universe.set_index("ticker")["sector"]
    missing = set(prices.columns) - set(sectors.index)
    assert not missing, f"tickers without a sector label: {missing}"

    factors = {k: pd.read_csv(p, index_col=0, parse_dates=True) for k, p in FACTOR_FILES.items()}
    raw_ls = pd.read_csv(f"{OUT_DIR}/long_short_returns.csv", index_col=0, parse_dates=True)
    raw_to = pd.read_csv(f"{OUT_DIR}/turnover.csv", index_col=0, parse_dates=True)

    # (a) sector-neutral decile sorts
    sn_ls, sn_to = {}, {}
    for name, factor in factors.items():
        sn_ls[name], sn_to[name] = decile_backtest(within_sector_percentile(factor, sectors), returns)
    sn_ls, sn_to = pd.DataFrame(sn_ls), pd.DataFrame(sn_to)
    sn_ls.to_csv(f"{OUT_DIR}/long_short_returns_sector_neutral.csv")
    sn_to.to_csv(f"{OUT_DIR}/turnover_sector_neutral.csv")

    raw_stats, sn_stats = spread_stats(raw_ls), spread_stats(sn_ls)
    sn_compare = pd.DataFrame({
        "raw_ann_ret_%": raw_stats["ann_ret_%"],
        "sn_ann_ret_%": sn_stats["ann_ret_%"],
        "raw_sharpe": raw_stats["sharpe"],
        "sn_sharpe": sn_stats["sharpe"],
        "raw_nw_t": raw_stats["nw_t"],
        "sn_nw_t": sn_stats["nw_t"],
        "sn_net_sharpe": spread_stats(net_returns(sn_ls, sn_to, COST_BPS))["sharpe"],
        "sn_turnover": sn_to.mean(),
        "corr_raw_sn": raw_ls.corrwith(sn_ls),
    })
    sn_compare.to_csv(f"{OUT_DIR}/sector_neutral_comparison.csv")

    # (a) sector-neutral Fama-MacBeth (sector dummies)
    sn_slopes, _, _ = cross_sectional_regressions(returns, factors, sectors=sectors)
    sn_slopes.to_csv(f"{OUT_DIR}/fm_slopes_sector_neutral.csv")
    raw_slopes = pd.read_csv(f"{OUT_DIR}/fm_slopes_multivariate.csv", index_col=0, parse_dates=True)
    fm_compare = pd.concat({
        "raw": time_series_test(raw_slopes.drop(columns="const"))[["mean_%", "nw_t"]],
        "sector_neutral": time_series_test(sn_slopes.drop(columns="const"))[["mean_%", "nw_t"]],
    }, axis=1)
    fm_compare.to_csv(f"{OUT_DIR}/fm_sector_neutral_comparison.csv")

    # (b) VIX regimes, lagged one month
    vix = pd.read_csv(VIX_PATH, index_col=0, parse_dates=True)["vix"]
    vix_lag = vix.resample("ME").last().shift(1)
    sample = raw_slopes.index
    regime = pd.qcut(vix_lag.loc[sample], 3, labels=REGIMES)
    cutoffs = vix_lag.loc[sample].quantile([1 / 3, 2 / 3])

    ls_regime = regime_table({k: raw_ls[k].loc[sample] for k in raw_ls}, regime)
    fm_regime = regime_table({k: raw_slopes[k] for k in factors}, regime)
    ls_regime.to_csv(f"{OUT_DIR}/regime_long_short.csv")
    fm_regime.to_csv(f"{OUT_DIR}/regime_fm_slopes.csv")

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 20)
    print("Sector-neutral vs raw decile long-short (gross, full sample):")
    print(sn_compare.astype(float).round(2))
    print("\nMultivariate FM, raw vs sector dummies (%/mo per 1 SD, NW t):")
    print(fm_compare.astype(float).round(3))
    print(f"\nVIX regimes on {len(sample)} months ({sample.min():%Y-%m} to {sample.max():%Y-%m}); "
          f"tercile cut-offs {cutoffs.iloc[0]:.1f} / {cutoffs.iloc[1]:.1f}")
    print("\nLong-short mean monthly return (%) by lagged-VIX tercile:")
    print(ls_regime.astype(float).round(2))
    print("\nMultivariate FM slope (%/mo) by lagged-VIX tercile:")
    print(fm_regime.astype(float).round(2))
