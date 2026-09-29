"""
Stage 6c: risk adjustment. Is each long-short spread more than market exposure?

(a) CAPM time-series regressions of each long-short return on the market
    excess return:  LS_t = alpha + beta * (R_m,t - rf_t) + e_t.
    The long-short is self-financing, so its return is already an excess
    return and needs no rf subtraction. Market = SPY (dividend-adjusted);
    rf_t = 13-week T-bill yield at the end of month t-1, divided by 12.
    Alpha is annualised (x12); t-stats are Newey-West (4 lags).
    Robustness: the same regression against the equal-weighted average of
    the study universe (survivorship-biased, so secondary only).

(b) Fama-MacBeth with each stock's trailing 252-day beta as a fifth
    characteristic, so each factor's premium is measured holding market
    beta fixed.

(c) CAPM regressions of the monthly FM slope series themselves, since a
    slope is the return of a factor-mimicking portfolio that can still
    carry market exposure even with beta as a characteristic control.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm

from decile_backtest import FACTOR_FILES, PRICES_PATH, monthly_returns
from fama_macbeth import cross_sectional_regressions, time_series_test
from validation_costs import COST_BPS, net_returns, newey_west_lags

MARKET_PATH = "data/market.csv"
BETA_PATH = "data/beta_winsorized.csv"
OUT_DIR = "output"
FACTORS = list(FACTOR_FILES)


def market_excess(index):
    market = pd.read_csv(MARKET_PATH, index_col=0, parse_dates=True)
    monthly = market.resample("ME").last()
    rm = monthly["spy"].pct_change(fill_method=None)
    rf = monthly["tbill_yield_pct"].shift(1) / 100 / 12
    return (rm - rf).reindex(index), rf.reindex(index)


def capm(ls, mkt):
    rows = {}
    for col in ls:
        d = pd.concat([ls[col], mkt], axis=1, keys=["y", "m"]).dropna()
        fit = sm.OLS(d["y"], sm.add_constant(d["m"])).fit(
            cov_type="HAC", cov_kwds={"maxlags": newey_west_lags(len(d))})
        resid_sd = fit.resid.std() * np.sqrt(12)
        rows[col] = {
            "months": len(d),
            "alpha_ann_%": fit.params["const"] * 1200,
            "alpha_nw_t": fit.tvalues["const"],
            "beta": fit.params["m"],
            "beta_nw_t": fit.tvalues["m"],
            "r2": fit.rsquared,
            "info_ratio": fit.params["const"] * 12 / resid_sd,
        }
    return pd.DataFrame(rows).T


if __name__ == "__main__":
    prices = pd.read_csv(PRICES_PATH, index_col=0, parse_dates=True)
    returns = monthly_returns(prices)
    ls = pd.read_csv(f"{OUT_DIR}/long_short_returns.csv", index_col=0, parse_dates=True)
    to = pd.read_csv(f"{OUT_DIR}/turnover.csv", index_col=0, parse_dates=True)
    sn = pd.read_csv(f"{OUT_DIR}/long_short_returns_sector_neutral.csv", index_col=0, parse_dates=True)

    mkt, rf = market_excess(ls.index)
    ew_excess = returns.mean(axis=1).reindex(ls.index) - rf
    print(f"Market excess return: {mkt.count()} months, mean {mkt.mean() * 1200:.2f}%/yr; "
          f"mean rf {rf.mean() * 1200:.2f}%/yr")

    tables = {
        "gross": capm(ls, mkt),
        "net": capm(net_returns(ls, to, COST_BPS), mkt),
        "sector_neutral": capm(sn, mkt),
        "pre_2023": capm(ls.loc[:"2022"], mkt),
        "from_2023": capm(ls.loc["2023":], mkt),
        "vs_ew_universe": capm(ls, ew_excess),
    }
    alphas = pd.concat(tables, names=["spec", "factor"])
    alphas.to_csv(f"{OUT_DIR}/capm_alphas.csv")

    # (b) Fama-MacBeth with beta as a control
    factors = {k: pd.read_csv(p, index_col=0, parse_dates=True) for k, p in FACTOR_FILES.items()}
    factors["beta"] = pd.read_csv(BETA_PATH, index_col=0, parse_dates=True)
    slopes, r2, n_obs = cross_sectional_regressions(returns, factors)
    slopes.to_csv(f"{OUT_DIR}/fm_slopes_with_beta.csv")
    fm_beta = time_series_test(slopes)
    fm_beta.to_csv(f"{OUT_DIR}/fm_summary_with_beta.csv")
    uni_beta, _, _ = cross_sectional_regressions(returns, {"beta": factors["beta"]})
    uni_beta = time_series_test(uni_beta[["beta"]])
    uni_beta.to_csv(f"{OUT_DIR}/fm_summary_beta_univariate.csv")

    # A characteristic control (beta as a regressor) is not the same as hedging
    # the market: each monthly slope is the return of a factor-mimicking
    # portfolio that can itself load on the market. So regress the slope
    # series on the market excess return too, with and without beta control.
    base_slopes = pd.read_csv(f"{OUT_DIR}/fm_slopes_multivariate.csv", index_col=0, parse_dates=True)
    slope_alphas = pd.concat({
        (spec, period): capm(s.drop(columns="const").loc[start:end], market_excess(s.index)[0])
        for spec, s in [("four_factor", base_slopes), ("with_beta", slopes)]
        for period, (start, end) in [("full", (None, None)), ("pre_2023", (None, "2022"))]
    }, names=["spec", "period", "factor"])
    slope_alphas.to_csv(f"{OUT_DIR}/fm_slope_alphas.csv")

    corr = []
    for d in slopes.index:
        m = pd.DataFrame({k: factors[k].loc[d] for k in factors}).dropna()
        corr.append(m.corr("spearman")["beta"])
    beta_corr = pd.concat(corr, axis=1).mean(axis=1).drop("beta")
    beta_corr.to_csv(f"{OUT_DIR}/beta_rank_correlations.csv")

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 20)
    print("\nCAPM alphas of long-short portfolios (alpha %/yr, NW t):")
    print(alphas.astype(float).round(2))
    print(f"\nFM with beta control: {len(slopes)} months, avg {n_obs.mean():.0f} stocks, avg R^2 {r2.mean():.3f}")
    print(fm_beta.astype(float).round(3))
    print("\nMarket-adjusted FM slopes (alpha %/yr per 1 SD, NW t):")
    print(slope_alphas[["alpha_ann_%", "alpha_nw_t", "beta", "beta_nw_t"]].astype(float).round(2))
    print("\nUnivariate beta premium:")
    print(uni_beta.astype(float).round(3))
    print("\nAverage cross-sectional rank correlation with beta:")
    print(beta_corr.round(2))
