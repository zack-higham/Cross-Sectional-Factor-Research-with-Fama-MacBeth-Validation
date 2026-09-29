"""
Stage 1c: download ~15 years of daily adjusted-close prices for the
top-200 universe. Adjusted close removes the artificial price jumps
caused by dividends and stock splits, so return calculations reflect
actual investment performance. Daily granularity is needed for the
60-day realised volatility factor even though rebalancing is monthly.
"""

import pandas as pd
import yfinance as yf

UNIVERSE_PATH = "data/sp500_top200.csv"
OUT_PATH = "data/prices.csv"
START_DATE = "2011-01-01"


def fetch_prices(tickers: list[str]) -> pd.DataFrame:
    data = yf.download(
        tickers,
        start=START_DATE,
        auto_adjust=True,
        group_by="column",
        threads=True,
        progress=True,
    )
    return data["Close"]


if __name__ == "__main__":
    universe = pd.read_csv(UNIVERSE_PATH)
    tickers = universe["ticker"].tolist()
    print(f"Downloading {len(tickers)} tickers from {START_DATE}...")

    prices = fetch_prices(tickers)

    missing = [t for t in tickers if t not in prices.columns]
    if missing:
        print(f"\n{len(missing)} tickers returned no data at all: {missing}")

    prices.to_csv(OUT_PATH)

    print(f"\nSaved {prices.shape[0]} rows x {prices.shape[1]} tickers to {OUT_PATH}")
    print(f"Date range: {prices.index.min().date()} to {prices.index.max().date()}")

    coverage = prices.notna().sum() / len(prices)
    partial = coverage[coverage < 0.9].sort_values()
    print(f"\n{len(partial)} tickers with < 90% date coverage (likely later IPO/listing):")
    print(partial)
