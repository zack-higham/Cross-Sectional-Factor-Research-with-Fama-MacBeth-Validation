"""
Stage 5: out-of-sample validation and transaction costs.

Out-of-sample split: one calendar split date for every factor, taken
from the common sample of the multivariate Fama-MacBeth regressions
(the first month where all four factors exist). The first 60% of those
months are in-sample, the remaining 40% out-of-sample. Nothing in this
project is fitted in-sample (the factor definitions and decile rules are
fixed in advance from the literature), so the split tests whether the
premia are stable over time, not whether a fitted model overfits.

Costs: each month the long-short portfolio pays COST_BPS per unit of
turnover (sum of absolute weight changes, from Stage 3). The first month
is charged the full initial build (turnover 2: $1 long + $1 short).
Break-even cost = the per-unit cost at which the mean net return is zero.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm

OUT_DIR = "output"
IS_FRACTION = 0.6
COST_BPS = 10
SENSITIVITY_BPS = [0, 5, 10, 20]
INITIAL_BUILD_TURNOVER = 2.0


def newey_west_lags(n_obs):
    return int(np.floor(4 * (n_obs / 100) ** (2 / 9)))


def nw_tstat(series):
    s = series.dropna()
    fit = sm.OLS(s.values, np.ones(len(s))).fit(
        cov_type="HAC", cov_kwds={"maxlags": newey_west_lags(len(s))})
    return fit.tvalues[0]


def spread_stats(ls):
    """Annualised stats for a table of monthly long-short returns."""
    return pd.DataFrame({
        "months": ls.count(),
        "ann_ret_%": ls.mean() * 12 * 100,
        "ann_vol_%": ls.std() * np.sqrt(12) * 100,
        "sharpe": ls.mean() / ls.std() * np.sqrt(12),
        "nw_t": ls.apply(nw_tstat),
    })


def fm_stats(slopes):
    return pd.DataFrame({
        "mean_%": slopes.mean() * 100,
        "nw_t": slopes.apply(nw_tstat),
        "months": slopes.count(),
    })


def net_returns(gross, turnover, cost_bps):
    """Charge cost_bps per unit turnover; first month charged the initial build."""
    to = turnover.reindex(gross.index)
    for col in to:
        first = gross[col].first_valid_index()
        to.loc[first, col] = INITIAL_BUILD_TURNOVER
    return gross - to.fillna(0) * cost_bps / 10_000


if __name__ == "__main__":
    ls = pd.read_csv(f"{OUT_DIR}/long_short_returns.csv", index_col=0, parse_dates=True)
    turnover = pd.read_csv(f"{OUT_DIR}/turnover.csv", index_col=0, parse_dates=True)
    slopes = pd.read_csv(f"{OUT_DIR}/fm_slopes_multivariate.csv", index_col=0, parse_dates=True)

    # split date from the common (multivariate FM) sample, applied to everything
    common = slopes.index
    split_date = common[int(len(common) * IS_FRACTION) - 1]
    ls_common = ls.loc[common.min():]
    periods = {
        "in_sample": (common.min(), split_date),
        "out_of_sample": (split_date + pd.offsets.MonthEnd(1), common.max()),
    }

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    for label, (start, end) in periods.items():
        print(f"{label}: {start:%Y-%m} to {end:%Y-%m}")

    split_rows = []
    for label, (start, end) in periods.items():
        spread = spread_stats(ls_common.loc[start:end]).add_prefix("ls_")
        fm = fm_stats(slopes.loc[start:end].drop(columns="const")).add_prefix("fm_")
        split_rows.append(pd.concat([spread, fm], axis=1).assign(period=label))
    split = pd.concat(split_rows).rename_axis("factor").reset_index().set_index(["period", "factor"])
    split.to_csv(f"{OUT_DIR}/is_oos_comparison.csv")
    print("\nIn-sample vs out-of-sample (long-short gross, and multivariate FM):")
    print(split.astype(float).round(2))

    # costs: full sample of each factor's long-short series
    gross = spread_stats(ls)
    net = spread_stats(net_returns(ls, turnover, COST_BPS))
    mean_to = turnover.mean()
    costs = pd.DataFrame({
        "gross_ann_ret_%": gross["ann_ret_%"],
        "net_ann_ret_%": net["ann_ret_%"],
        "gross_sharpe": gross["sharpe"],
        "net_sharpe": net["sharpe"],
        "net_nw_t": net["nw_t"],
        "mean_turnover": mean_to,
        "ann_cost_%": mean_to * 12 * COST_BPS / 100,
        # undefined when the gross spread is already negative
        "breakeven_bps": (ls.mean() / mean_to * 10_000).where(ls.mean() > 0),
    })
    costs.to_csv(f"{OUT_DIR}/costs_summary.csv")
    print(f"\nGross vs net of {COST_BPS}bps per unit turnover (full sample):")
    print(costs.round(2))

    sens = pd.DataFrame({f"{c}bps": spread_stats(net_returns(ls, turnover, c))["sharpe"]
                         for c in SENSITIVITY_BPS})
    sens.to_csv(f"{OUT_DIR}/cost_sensitivity_sharpe.csv")
    print("\nNet Sharpe by cost assumption:")
    print(sens.round(2))
