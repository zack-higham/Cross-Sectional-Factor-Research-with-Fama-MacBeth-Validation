"""
Stage 2a: 12-1 momentum factor. For each stock, each month, the
cumulative return over the 12-month window ending 1 month ago -
skipping the most recent month to avoid mixing in short-term reversal.
"""

import pandas as pd

PRICES_PATH = "Project-1/data/prices.csv"
OUT_PATH = "Project-1/data/momentum.csv"

if __name__ == "__main__":
    prices = pd.read_csv(PRICES_PATH, index_col=0, parse_dates=True)

    monthly_prices = prices.resample("ME").last()
    print(f"Resampled to {len(monthly_prices)} month-end rows")

    momentum = monthly_prices.shift(1) / monthly_prices.shift(13) - 1
    momentum.to_csv(OUT_PATH)

    non_null_counts = momentum.notna().sum(axis=1)
    print("\nNumber of stocks with a valid momentum value, by month (first 20 non-empty months):")
    print(non_null_counts[non_null_counts > 0].head(20))

    print("\nCross-sectional momentum stats for the most recent month:")
    print(momentum.iloc[-1].describe())
