"""
Risk control: trailing market beta. For each stock, the OLS slope of its
daily returns on SPY's daily returns over the last 252 trading days
(about one year), beta = cov(r_i, r_m) / var(r_m). As with volatility,
the window is measured at month-end and then shifted one month, so the
value paired with month t's return uses data only up to the end of t-1.

Beta is not one of the studied factors: it is the control that shows
whether a factor spread is anything more than exposure to the market.
"""

import pandas as pd

PRICES_PATH = "data/prices.csv"
MARKET_PATH = "data/market.csv"
OUT_PATH = "data/beta.csv"
WINDOW = 252

if __name__ == "__main__":
    prices = pd.read_csv(PRICES_PATH, index_col=0, parse_dates=True)
    spy = pd.read_csv(MARKET_PATH, index_col=0, parse_dates=True)["spy"]

    stock_ret = prices.pct_change(fill_method=None)
    mkt_ret = spy.pct_change(fill_method=None).reindex(stock_ret.index)

    # rolling moments; a window only counts once all 252 days are present
    xy = stock_ret.mul(mkt_ret, axis=0).rolling(WINDOW).mean()
    x = stock_ret.rolling(WINDOW).mean()
    y = mkt_ret.rolling(WINDOW).mean()
    var_m = (mkt_ret ** 2).rolling(WINDOW).mean() - y ** 2
    beta_daily = (xy - x.mul(y, axis=0)).div(var_m, axis=0)

    beta = beta_daily.resample("ME").last().shift(1)
    beta.to_csv(OUT_PATH)

    latest = beta.dropna(how="all").iloc[-1].dropna()
    print(f"First valid month {beta.dropna(how='all').index.min():%Y-%m}; latest month {len(latest)} stocks")
    print(f"Latest cross-section: mean {latest.mean():.2f}, median {latest.median():.2f}, "
          f"min {latest.min():.2f} ({latest.idxmin()}), max {latest.max():.2f} ({latest.idxmax()})")
