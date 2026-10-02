"""
Robustness extension 1: maximum drawdown and Calmar ratio of every
long-short portfolio (raw and sector-neutral, gross and net of costs).
Specification fixed in robustness_spec.py before this was run.

Wealth compounds the monthly long-short return from W_0 = 1 in the month
before the first return, so a loss in the very first month counts as a
drawdown. Drawdown at t is W_t / (running peak of W) - 1. The recovery
month is the first month after the trough in which wealth is back at the
pre-drawdown peak. Calmar is the annualised arithmetic mean return (the
same "Ann. return" as the paper's other tables) divided by |max drawdown|.

Monthly returns only: a drawdown that opens and closes within a month is
invisible here, so these figures understate intra-month drawdowns.
"""

import pandas as pd

from robustness_spec import ANNUALISE, COST_BPS
from validation_costs import net_returns

OUT_DIR = "output"


def max_drawdown(r):
    """Max drawdown (%), peak/trough/recovery months for one monthly series."""
    r = r.dropna()
    start = r.index[0] - pd.offsets.MonthEnd(1)
    wealth = pd.concat([pd.Series([1.0], index=[start]), (1 + r).cumprod()])
    peak = wealth.cummax()
    dd = wealth / peak - 1
    trough = dd.idxmin()
    peak_date = wealth.loc[:trough].idxmax()
    after = wealth.loc[trough:]
    recovered = after[after >= wealth.loc[peak_date]]
    return {
        "mdd_%": dd.min() * 100,
        "peak": peak_date,
        "trough": trough,
        "recovery": recovered.index[0] if len(recovered) else pd.NaT,
        "months_peak_to_trough": len(wealth.loc[peak_date:trough]) - 1,
        "worst_month_%": r.min() * 100,
        "worst_month": r.idxmin(),
    }


def drawdown_table(ls):
    rows = {}
    for col in ls:
        row = max_drawdown(ls[col])
        row["ann_ret_%"] = ls[col].mean() * ANNUALISE * 100
        row["calmar"] = row["ann_ret_%"] / abs(row["mdd_%"])
        rows[col] = row
    return pd.DataFrame(rows).T


if __name__ == "__main__":
    load = lambda f: pd.read_csv(f"{OUT_DIR}/{f}", index_col=0, parse_dates=True)
    raw, raw_to = load("long_short_returns.csv"), load("turnover.csv")
    sn, sn_to = load("long_short_returns_sector_neutral.csv"), load("turnover_sector_neutral.csv")

    table = pd.concat({
        ("raw", "gross"): drawdown_table(raw),
        ("raw", "net"): drawdown_table(net_returns(raw, raw_to, COST_BPS)),
        ("sector_neutral", "gross"): drawdown_table(sn),
        ("sector_neutral", "net"): drawdown_table(net_returns(sn, sn_to, COST_BPS)),
    }, names=["construction", "costs", "factor"])
    table.to_csv(f"{OUT_DIR}/drawdowns.csv")

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 20)
    show = table.copy()
    for c in ["peak", "trough", "recovery", "worst_month"]:
        show[c] = pd.to_datetime(show[c]).dt.strftime("%Y-%m").fillna("not recovered")
    for c in ["mdd_%", "worst_month_%", "ann_ret_%", "calmar"]:
        show[c] = show[c].astype(float).round(2)
    print(show)

    # pre-registered checks
    mom = table.loc[("raw", "gross", "momentum")]
    assert mom["mdd_%"] <= mom["worst_month_%"], "MDD cannot be smaller than the worst single month"
    size = table.loc[("raw", "gross", "size")]
    print(f"\nCheck: momentum raw gross MDD {mom['mdd_%']:.1f}% vs worst month {mom['worst_month_%']:.1f}%")
    print(f"Check: size raw gross MDD {size['mdd_%']:.1f}%, recovery: {size['recovery']}")
