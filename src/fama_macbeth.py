"""
Stage 4: Fama-MacBeth (1973) regressions.

Step 1: each month t, run a cross-sectional OLS of stock returns earned
during month t on the factor values known at the start of month t:

    r_{i,t} = a_t + sum_k g_{k,t} * z_{k,i,t} + e_{i,t}

Step 2: treat the monthly slopes g_{k,t} as a time series and test
whether their mean is zero. The t-stat uses Newey-West (HAC) standard
errors, because the monthly slopes can be autocorrelated (momentum and
size change slowly month to month, so consecutive cross-sections are
not independent draws).

Factors are z-scored cross-sectionally each month, so each slope is the
monthly return premium per one cross-sectional standard deviation of
that factor, comparable across factors and across months. Timing is the
same as Stage 3: factor row t uses prices up to end of month t-1 and is
paired with the return over month t.

Two specifications are reported:
  - univariate: one factor per regression (comparable to the decile spreads)
  - multivariate: all four factors jointly, on the common sample of
    stocks with every factor available (each slope controls for the others)
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm

PRICES_PATH = "Project-1/data/prices.csv"
FACTOR_FILES = {
    "momentum": "Project-1/data/momentum_winsorized.csv",
    "reversal": "Project-1/data/reversal_winsorized.csv",
    "volatility": "Project-1/data/volatility_winsorized.csv",
    "size": "Project-1/data/size.csv",
}
OUT_DIR = "Project-1/output"
MIN_STOCKS = 100  # same threshold as the decile backtest


def newey_west_lags(n_obs):
    """Newey-West (1994) plug-in lag rule: floor(4 * (T/100)^(2/9))."""
    return int(np.floor(4 * (n_obs / 100) ** (2 / 9)))


def cross_sectional_regressions(returns, factors, sectors=None):
    """Step 1: one OLS per month. Returns slopes (T x K+1), R^2 and N per month.

    If `sectors` (ticker -> sector label) is given, sector dummies are added,
    so each factor slope is identified only from within-sector variation.
    Dummy coefficients are dropped from the returned slopes.
    """
    names = list(factors)
    slopes, r2, n_obs = {}, {}, {}

    for date in returns.index:
        if not all(date in factors[k].index for k in names):
            continue
        month = pd.DataFrame({k: factors[k].loc[date] for k in names})
        month["ret"] = returns.loc[date]
        month = month.dropna()
        if len(month) < MIN_STOCKS:
            continue

        # z-score on the regression sample so every slope is per 1 SD
        # of the stocks actually in this month's regression
        x = (month[names] - month[names].mean()) / month[names].std()
        x = sm.add_constant(x)
        if sectors is not None:
            dummies = pd.get_dummies(sectors.reindex(month.index), drop_first=True, dtype=float)
            x = pd.concat([x, dummies], axis=1)
        fit = sm.OLS(month["ret"], x).fit()
        slopes[date] = fit.params[["const"] + names]
        r2[date] = fit.rsquared
        n_obs[date] = len(month)

    return pd.DataFrame(slopes).T, pd.Series(r2), pd.Series(n_obs)


def time_series_test(slopes):
    """Step 2: mean slope, plain FM t-stat, and Newey-West t-stat."""
    rows = {}
    for col in slopes.columns:
        g = slopes[col].dropna()
        lags = newey_west_lags(len(g))
        nw = sm.OLS(g.values, np.ones(len(g))).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
        rows[col] = {
            "mean_%": g.mean() * 100,
            "fm_t": g.mean() / (g.std() / np.sqrt(len(g))),
            "nw_t": nw.tvalues[0],
            "nw_p": nw.pvalues[0],
            "nw_lags": lags,
            "pct_months_pos": (g > 0).mean() * 100,
            "months": len(g),
        }
    return pd.DataFrame(rows).T


if __name__ == "__main__":
    prices = pd.read_csv(PRICES_PATH, index_col=0, parse_dates=True)
    monthly_prices = prices.resample("ME").last()
    returns = monthly_prices.pct_change(fill_method=None)

    factors = {k: pd.read_csv(p, index_col=0, parse_dates=True) for k, p in FACTOR_FILES.items()}

    # univariate: each factor on its own
    uni_rows = []
    for name in factors:
        slopes, _, _ = cross_sectional_regressions(returns, {name: factors[name]})
        uni_rows.append(time_series_test(slopes[[name]]))
    univariate = pd.concat(uni_rows)

    # multivariate: all factors jointly
    slopes, r2, n_obs = cross_sectional_regressions(returns, factors)
    multivariate = time_series_test(slopes)

    slopes.to_csv(f"{OUT_DIR}/fm_slopes_multivariate.csv")
    univariate.to_csv(f"{OUT_DIR}/fm_summary_univariate.csv")
    multivariate.to_csv(f"{OUT_DIR}/fm_summary_multivariate.csv")

    pd.set_option("display.width", 200)
    print(f"Multivariate sample: {len(slopes)} months, "
          f"{slopes.index.min():%Y-%m} to {slopes.index.max():%Y-%m}, "
          f"avg {n_obs.mean():.0f} stocks/month, avg R^2 {r2.mean():.3f}")
    print("\nUnivariate Fama-MacBeth (slope = monthly return per 1 SD of factor):")
    print(univariate.astype(float).round(3))
    print("\nMultivariate Fama-MacBeth:")
    print(multivariate.astype(float).round(3))
