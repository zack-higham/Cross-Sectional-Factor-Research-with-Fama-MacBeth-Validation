"""
Risk-adjustment inputs: daily SPY (dividend-adjusted, a total-return
proxy for the S&P 500) and the 13-week T-bill yield (^IRX, annualised %).
SPY is the market benchmark rather than an average of the study universe,
because SPY holds the index's constituents as they were at the time, so
it does not share the universe's survivorship bias.
"""

import pandas as pd
import yfinance as yf

OUT_PATH = "data/market.csv"
START_DATE = "2010-01-01"  # one extra year so trailing 252-day betas exist from early 2011

if __name__ == "__main__":
    spy = yf.download("SPY", start=START_DATE, auto_adjust=True, progress=False)["Close"].squeeze()
    irx = yf.download("^IRX", start=START_DATE, auto_adjust=False, progress=False)["Close"].squeeze()
    market = pd.DataFrame({"spy": spy, "tbill_yield_pct": irx})
    market.to_csv(OUT_PATH)
    print(f"Saved {len(market)} rows, {market.index.min():%Y-%m-%d} to {market.index.max():%Y-%m-%d}")
    print(market.describe().round(2))
