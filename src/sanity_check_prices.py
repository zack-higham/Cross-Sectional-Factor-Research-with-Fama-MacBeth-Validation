"""
Stage 1d: sanity-check the downloaded price data before any factor is
built on top of it. A single bad data point here (a failed download,
a missing day mid-history, an unadjusted split) would silently corrupt
momentum/volatility calculations later with no error raised.
"""

import pandas as pd

PRICES_PATH = "Project-1/data/prices.csv"
EXTREME_MOVE_THRESHOLD = 0.5  # 50% single-day move

if __name__ == "__main__":
    prices = pd.read_csv(PRICES_PATH, index_col=0, parse_dates=True)
    print(f"Shape: {prices.shape[0]} dates x {prices.shape[1]} tickers")

    # 1. Tickers with zero data at all
    empty = prices.columns[prices.isna().all()]
    print(f"\n[1] Tickers with zero data: {len(empty)}")
    if len(empty):
        print(list(empty))

    # 2. Coverage per ticker (share of dates with a price)
    coverage = prices.notna().mean().sort_values()
    low_coverage = coverage[coverage < 0.9]
    print(f"\n[2] Tickers with < 90% date coverage: {len(low_coverage)}")
    print(low_coverage)

    # 3. Interior gaps: missing values between a ticker's first and
    # last valid date. A gap before listing or after delisting is
    # normal; a gap in the middle of active trading is not.
    print("\n[3] Interior gaps (missing days between first and last valid date):")
    interior_gap_counts = {}
    for ticker in prices.columns:
        series = prices[ticker]
        first, last = series.first_valid_index(), series.last_valid_index()
        if first is None:
            continue
        window = series.loc[first:last]
        gaps = window.isna().sum()
        if gaps > 0:
            interior_gap_counts[ticker] = gaps

    if interior_gap_counts:
        print(f"{len(interior_gap_counts)} tickers affected:")
        print(pd.Series(interior_gap_counts).sort_values(ascending=False))
    else:
        print("None found.")

    # 4. Extreme single-day moves: real volatility, or a bad
    # tick / unadjusted split that would distort factors if left in.
    daily_returns = prices.pct_change()
    extreme = daily_returns[daily_returns.abs() > EXTREME_MOVE_THRESHOLD].stack()
    print(f"\n[4] Single-day moves > {EXTREME_MOVE_THRESHOLD:.0%}: {len(extreme)}")
    if len(extreme):
        print(extreme.sort_values(key=abs, ascending=False).head(15))
