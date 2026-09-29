"""
Stage 2b: short-term reversal factor - prior 1-month return. Uses the
exact window momentum deliberately excluded, so the two factors are
built from adjacent, non-overlapping slices of the same price history.
"""

import pandas as pd

PRICES_PATH = "data/prices.csv"
OUT_PATH = "data/reversal.csv"

if __name__ == "__main__":
    prices = pd.read_csv(PRICES_PATH, index_col=0, parse_dates=True)
    monthly_prices = prices.resample("ME").last()

    reversal = monthly_prices.shift(1) / monthly_prices.shift(2) - 1
    reversal.to_csv(OUT_PATH)

    non_null_counts = reversal.notna().sum(axis=1)
    print("Stocks with a valid reversal value, by month (first 10 non-empty months):")
    print(non_null_counts[non_null_counts > 0].head(10))

    print("\nCross-sectional reversal stats for the most recent month:")
    print(reversal.iloc[-1].describe())
