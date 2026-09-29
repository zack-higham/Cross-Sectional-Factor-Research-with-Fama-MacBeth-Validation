"""
Stage 7a: information coefficient (IC). Each month, the Spearman rank
correlation across stocks between the factor value known at the start
of month t and the return earned during month t (same timing as the
decile sorts and Fama-MacBeth). Rank correlation is used so the IC
measures how well the factor *orders* next-month returns, insensitive
to outliers in either variable.

Reported: mean IC, IC volatility, ICIR (mean / std, annualised by
sqrt(12)), Newey-West t-stat on the mean, and % of months with IC > 0.
"""

import numpy as np
import pandas as pd

from decile_backtest import FACTOR_FILES, MIN_STOCKS, PRICES_PATH, monthly_returns
from validation_costs import nw_tstat

OUT_DIR = "Project-1/output"


def monthly_ic(factor, returns):
    rows = {}
    for date in factor.index.intersection(returns.index):
        pair = pd.DataFrame({"f": factor.loc[date], "r": returns.loc[date]}).dropna()
        if len(pair) >= MIN_STOCKS:
            rows[date] = pair["f"].corr(pair["r"], method="spearman")
    return pd.Series(rows)


if __name__ == "__main__":
    prices = pd.read_csv(PRICES_PATH, index_col=0, parse_dates=True)
    returns = monthly_returns(prices)

    ic = pd.DataFrame({k: monthly_ic(pd.read_csv(p, index_col=0, parse_dates=True), returns)
                       for k, p in FACTOR_FILES.items()})
    ic.to_csv(f"{OUT_DIR}/ic_monthly.csv")

    summary = pd.DataFrame({
        "months": ic.count(),
        "mean_ic": ic.mean(),
        "ic_std": ic.std(),
        "icir_ann": ic.mean() / ic.std() * np.sqrt(12),
        "nw_t": ic.apply(nw_tstat),
        "pct_pos": (ic > 0).sum() / ic.count() * 100,  # denominator excludes warm-up NaNs
    })
    summary.to_csv(f"{OUT_DIR}/ic_summary.csv")
    print(summary.round(3))
