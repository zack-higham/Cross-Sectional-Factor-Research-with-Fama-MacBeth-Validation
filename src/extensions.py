"""
Stage 6: sector-neutral factors.

The universe has sectors with as few as 5 stocks, too few for
within-sector deciles. Instead each stock's factor is converted to a
percentile rank *within its GICS sector* each month, and deciles are
formed on that pooled within-sector percentile. Every decile then holds a
similar sector mix, so the long-short spread is close to sector-neutral.
The Fama-MacBeth counterpart adds sector dummies, so slopes use only
within-sector variation.

Sector labels are today's GICS classification, applied to the whole
history (a stated limitation: e.g. the 2018 GICS reshuffle that created
Communication Services is not reflected historically).
"""

import pandas as pd

from decile_backtest import FACTOR_FILES, N_DECILES, PRICES_PATH, assign_deciles, long_short_weights, monthly_returns
from fama_macbeth import cross_sectional_regressions, time_series_test
from validation_costs import COST_BPS, net_returns, spread_stats

UNIVERSE_PATH = "data/sp500_top200.csv"
OUT_DIR = "output"


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


if __name__ == "__main__":
    prices = pd.read_csv(PRICES_PATH, index_col=0, parse_dates=True)
    returns = monthly_returns(prices)

    universe = pd.read_csv(UNIVERSE_PATH)
    sectors = universe.set_index("ticker")["sector"]
    missing = set(prices.columns) - set(sectors.index)
    assert not missing, f"tickers without a sector label: {missing}"

    factors = {k: pd.read_csv(p, index_col=0, parse_dates=True) for k, p in FACTOR_FILES.items()}
    raw_ls = pd.read_csv(f"{OUT_DIR}/long_short_returns.csv", index_col=0, parse_dates=True)
    raw_to = pd.read_csv(f"{OUT_DIR}/turnover.csv", index_col=0, parse_dates=True)

    # sector-neutral decile sorts
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

    # sector-neutral Fama-MacBeth (sector dummies)
    sn_slopes, _, _ = cross_sectional_regressions(returns, factors, sectors=sectors)
    sn_slopes.to_csv(f"{OUT_DIR}/fm_slopes_sector_neutral.csv")
    raw_slopes = pd.read_csv(f"{OUT_DIR}/fm_slopes_multivariate.csv", index_col=0, parse_dates=True)
    fm_compare = pd.concat({
        "raw": time_series_test(raw_slopes.drop(columns="const"))[["mean_%", "nw_t"]],
        "sector_neutral": time_series_test(sn_slopes.drop(columns="const"))[["mean_%", "nw_t"]],
    }, axis=1)
    fm_compare.to_csv(f"{OUT_DIR}/fm_sector_neutral_comparison.csv")

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 20)
    print("Sector-neutral vs raw decile long-short (gross, full sample):")
    print(sn_compare.astype(float).round(2))
    print("\nMultivariate FM, raw vs sector dummies (%/mo per 1 SD, NW t):")
    print(fm_compare.astype(float).round(3))
