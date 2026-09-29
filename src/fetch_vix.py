"""
Stage 6b input: daily CBOE VIX close, same start date as the price data.
Used only as a regime-conditioning variable (market-implied volatility),
never as a factor.
"""

import yfinance as yf

OUT_PATH = "Project-1/data/vix.csv"
START_DATE = "2011-01-01"

if __name__ == "__main__":
    data = yf.download("^VIX", start=START_DATE, auto_adjust=False, progress=False)
    vix = data["Close"].squeeze().rename("vix")
    vix.to_csv(OUT_PATH)
    print(f"Saved {len(vix)} daily VIX closes, {vix.index.min():%Y-%m-%d} to "
          f"{vix.index.max():%Y-%m-%d}, range {vix.min():.1f} to {vix.max():.1f}")
