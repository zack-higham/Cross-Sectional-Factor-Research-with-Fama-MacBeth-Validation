"""
Robustness extension 2: alphas against the Fama-French factor models
with momentum. Specification fixed in robustness_spec.py before this was
run.

Test series: the four long-short portfolios (gross and net of 10 bps)
and the four multivariate Fama-MacBeth slope series. Each is regressed
on the French factor returns of the same calendar month (the long-short
return for month t is earned during month t, as are the factor returns),
so no lag is applied:

    y_t = alpha + sum_j b_j * F_{j,t} + e_t

for CAPM, FF3, Carhart (FF3 + UMD) and FF5 + UMD. Alphas are annualised
(x12) with Newey-West t-statistics, lag rule as in the rest of the paper.

The French factors are built from the full CRSP universe, so they are
free of survivorship bias; the test portfolios are not. A loading on SMB
therefore measures co-movement with the true size factor, while the
alpha absorbs whatever this universe's construction adds.

Alignment check: the paper's SPY excess return and French Mkt-RF measure
the same thing, so if their monthly correlation is not above 0.95 the
months are misaligned and the script stops.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm

from robustness_spec import COST_BPS, MIN_ALIGNMENT_CORR, MODELS, PERIODS
from risk_adjustment import market_excess
from validation_costs import net_returns, newey_west_lags

OUT_DIR = "output"
FRENCH_PATH = "data/french_factors.csv"
FACTORS = ["momentum", "reversal", "volatility", "size"]


def factor_regression(y, X):
    d = pd.concat([y.rename("y"), X], axis=1).dropna()
    fit = sm.OLS(d["y"], sm.add_constant(d[X.columns])).fit(
        cov_type="HAC", cov_kwds={"maxlags": newey_west_lags(len(d))})
    row = {"months": len(d), "start": d.index.min(), "end": d.index.max(),
           "alpha_ann_%": fit.params["const"] * 1200, "alpha_nw_t": fit.tvalues["const"],
           "r2": fit.rsquared}
    for f in X.columns:
        row[f"b_{f}"] = fit.params[f]
        row[f"t_{f}"] = fit.tvalues[f]
    return row


def run_all(series, ff):
    rows = {}
    for name, y in series.items():
        for model, cols in MODELS.items():
            for period, (start, end) in PERIODS.items():
                rows[(name, model, period)] = factor_regression(y.loc[start:end], ff[cols])
    out = pd.DataFrame(rows).T
    out.index.names = ["series", "model", "period"]
    return out


if __name__ == "__main__":
    load = lambda f: pd.read_csv(f"{OUT_DIR}/{f}", index_col=0, parse_dates=True)
    ls, to = load("long_short_returns.csv"), load("turnover.csv")
    slopes = load("fm_slopes_multivariate.csv")[FACTORS]
    ff = pd.read_csv(FRENCH_PATH, index_col=0, parse_dates=True)

    # alignment check against the paper's own market series
    spy_excess, _ = market_excess(ls.index)
    both = pd.concat([spy_excess, ff["Mkt-RF"]], axis=1, join="inner").dropna()
    corr = both.corr().iloc[0, 1]
    lagged = both.iloc[:, 0].corr(both.iloc[:, 1].shift(1))
    print(f"SPY excess vs French Mkt-RF: {len(both)} months {both.index.min():%Y-%m} to "
          f"{both.index.max():%Y-%m}, corr {corr:.4f} (one-month misaligned: {lagged:.4f}); "
          f"mean {both.mean().iloc[0] * 1200:.2f} vs {both.mean().iloc[1] * 1200:.2f} %/yr")
    assert corr > MIN_ALIGNMENT_CORR, "factor months are misaligned"
    pd.Series({"months": len(both), "corr": corr, "corr_misaligned_1m": lagged,
               "french_end": f"{ff.index.max():%Y-%m}"}).to_csv(f"{OUT_DIR}/french_alignment.csv")
    print(f"French data end {ff.index.max():%Y-%m}; long-short data end {ls.index.max():%Y-%m}")

    net = net_returns(ls, to, COST_BPS)
    series = {**{f"{f}_gross": ls[f] for f in FACTORS},
              **{f"{f}_net": net[f] for f in FACTORS},
              **{f"{f}_fm": slopes[f] for f in FACTORS}}
    table = run_all(series, ff)
    table.to_csv(f"{OUT_DIR}/multifactor_alphas.csv")

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.max_rows", 200)
    num_cols = [c for c in table.columns if c not in ("start", "end")]
    show = table[num_cols].astype(float).round(2)
    print("\nFull sample, all models:")
    print(show.xs("full", level="period")[["months", "alpha_ann_%", "alpha_nw_t", "r2", "b_Mkt-RF",
                                            "b_SMB", "b_HML", "b_RMW", "b_CMA", "b_UMD", "t_UMD", "t_SMB"]])
    print("\nAlpha (NW t) by period:")
    print(show[["months", "alpha_ann_%", "alpha_nw_t"]].unstack("period").round(2))

    # consistency with the paper's SPY-based CAPM (capm_alphas.csv, gross)
    capm_spy = pd.read_csv(f"{OUT_DIR}/capm_alphas.csv", index_col=[0, 1]).loc["gross"]
    capm_ff = table.xs(("CAPM", "full"), level=["model", "period"]).loc[[f"{f}_gross" for f in FACTORS]]
    print("\nCAPM gross: SPY-based alpha/beta vs French Mkt-RF alpha/beta")
    print(pd.DataFrame({"spy_alpha": capm_spy["alpha_ann_%"].values, "ff_alpha": capm_ff["alpha_ann_%"].astype(float).values,
                        "spy_beta": capm_spy["beta"].values, "ff_beta": capm_ff["b_Mkt-RF"].astype(float).values},
                       index=FACTORS).round(2))
