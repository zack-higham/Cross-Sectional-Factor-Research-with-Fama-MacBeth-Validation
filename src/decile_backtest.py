"""
Stage 3: decile portfolio backtest. Each month, for each factor, rank
the cross-section into 10 equal-count buckets (D1 = lowest factor
value, D10 = highest), hold each bucket equal-weighted for that month,
and record its return. The long-short spread is D10 - D1.

Timing: factor row t uses only prices up to end of month t-1 (see the
factor scripts), so it is paired with the return earned *during*
month t: P(t) / P(t-1) - 1. Same row, no extra shift needed here.

Sign convention: always D10 - D1 (high minus low), for every factor -
so a negative spread for reversal/volatility is the "expected" result,
and the sign lines up with the Stage 4 Fama-MacBeth coefficients.

Turnover: sum of |w_t - w_{t-1}| over the long-short weights, with
weights reset to equal each month (ignores intra-month price drift -
a stated simplification). This is total traded notional per $1 long /
$1 short, which is what Stage 5's 10bps cost multiplies.
"""

import numpy as np
import pandas as pd

PRICES_PATH = "Project-1/data/prices.csv"
FACTOR_FILES = {
    "momentum": "Project-1/data/momentum_winsorized.csv",
    "reversal": "Project-1/data/reversal_winsorized.csv",
    "volatility": "Project-1/data/volatility_winsorized.csv",
    "size": "Project-1/data/size.csv",
}
OUT_DIR = "Project-1/output"
N_DECILES = 10
MIN_STOCKS = 100  # skip months with fewer than ~10 stocks per decile


def monthly_returns(prices):
    """Month-end to month-end returns. A final month whose last price is
    before that month's last business day is incomplete (a partial-month
    return mixed in with full months), so it is dropped."""
    monthly_prices = prices.resample("ME").last()
    # fill_method=None: never forward-fill a missing price into a fake 0% return
    returns = monthly_prices.pct_change(fill_method=None)
    if prices.index[-1] < prices.index[-1] + pd.offsets.BMonthEnd(0):
        returns = returns.iloc[:-1]
    return returns


def assign_deciles(factor):
    """Per-month (row-wise) percentile rank -> decile label 1..10."""
    pct_rank = factor.rank(axis=1, pct=True)
    deciles = np.ceil(pct_rank * N_DECILES)
    enough_stocks = factor.notna().sum(axis=1) >= MIN_STOCKS
    return deciles.where(enough_stocks, axis=0)


def long_short_weights(deciles):
    """+1/n on D10 names, -1/n on D1 names, 0 elsewhere."""
    long_leg = (deciles == N_DECILES).astype(float)
    short_leg = (deciles == 1).astype(float)
    return long_leg.div(long_leg.sum(axis=1), axis=0) - short_leg.div(short_leg.sum(axis=1), axis=0)


if __name__ == "__main__":
    prices = pd.read_csv(PRICES_PATH, index_col=0, parse_dates=True)
    returns = monthly_returns(prices)

    decile_returns = {}
    long_short = {}
    turnover = {}

    for name, path in FACTOR_FILES.items():
        factor = pd.read_csv(path, index_col=0, parse_dates=True)
        deciles = assign_deciles(factor)

        decile_returns[name] = pd.DataFrame({
            d: returns.where(deciles == d).mean(axis=1) for d in range(1, N_DECILES + 1)
        }).dropna(how="all")

        long_short[name] = decile_returns[name][N_DECILES] - decile_returns[name][1]

        weights = long_short_weights(deciles).loc[long_short[name].index]
        turnover[name] = weights.diff().abs().sum(axis=1).iloc[1:]  # first month = initial build, not turnover

    decile_df = pd.concat(decile_returns, axis=1)
    ls_df = pd.DataFrame(long_short)
    to_df = pd.DataFrame(turnover)
    decile_df.to_csv(f"{OUT_DIR}/decile_returns.csv")
    ls_df.to_csv(f"{OUT_DIR}/long_short_returns.csv")
    to_df.to_csv(f"{OUT_DIR}/turnover.csv")

    print("Mean monthly return by decile (%):")
    print((pd.DataFrame({n: decile_returns[n].mean() for n in FACTOR_FILES}).T * 100).round(2))

    print("\nLong-short (D10 - D1) summary, gross of costs:")
    summary = pd.DataFrame({
        "months": ls_df.count(),
        "mean_mo_%": ls_df.mean() * 100,
        "ann_ret_%": ls_df.mean() * 12 * 100,
        "ann_vol_%": ls_df.std() * np.sqrt(12) * 100,
        "sharpe": ls_df.mean() / ls_df.std() * np.sqrt(12),
        "t_stat": ls_df.mean() / (ls_df.std() / np.sqrt(ls_df.count())),
        "turnover": to_df.mean(),
        "per_leg_replaced_%": to_df.mean() / 4 * 100,
    })
    print(summary.round(2))
