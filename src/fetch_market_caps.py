"""
Stage 1b: rank the S&P 500 constituent list by current market cap and
keep the largest 200. yfinance has no bulk market-cap endpoint, so this
loops one ticker at a time via fast_info, which is slow and occasionally
rate-limited/missing data — failures are logged and skipped rather than
allowed to kill the run.
"""

import time

import pandas as pd
import yfinance as yf

CONSTITUENTS_PATH = "Project-1/data/sp500_constituents.csv"
OUT_PATH = "Project-1/data/sp500_top200.csv"
TOP_N = 200
PAUSE_SECONDS = 0.1  # small delay between requests to reduce rate-limit risk


def fetch_market_caps(tickers: list[str]) -> pd.DataFrame:
    records = []
    failed = []

    for i, ticker in enumerate(tickers):
        try:
            market_cap = yf.Ticker(ticker).fast_info["marketCap"]
            records.append({"ticker": ticker, "market_cap": market_cap})
        except Exception as e:
            failed.append((ticker, str(e)))

        if (i + 1) % 50 == 0:
            print(f"  ...{i + 1}/{len(tickers)} tickers processed")

        time.sleep(PAUSE_SECONDS)

    if failed:
        print(f"\n{len(failed)} tickers failed and were skipped:")
        for ticker, err in failed[:10]:
            print(f"  {ticker}: {err}")
        if len(failed) > 10:
            print(f"  ... and {len(failed) - 10} more")

    return pd.DataFrame(records)


if __name__ == "__main__":
    constituents = pd.read_csv(CONSTITUENTS_PATH)
    print(f"Pulling market caps for {len(constituents)} tickers...")

    caps = fetch_market_caps(constituents["ticker"].tolist())

    merged = constituents.merge(caps, on="ticker", how="inner")
    top200 = merged.sort_values("market_cap", ascending=False).head(TOP_N)
    top200.to_csv(OUT_PATH, index=False)

    print(f"\nKept top {len(top200)} by market cap (out of {len(merged)} with data)")
    print(top200[["ticker", "name", "sector", "market_cap"]].head(10))
