"""
Stage 2c: trailing 60-day realised volatility factor. Computed from
daily returns (a rolling std dev needs many data points to be a
reliable estimate), then resampled to one value per month, then
lagged an extra month - unlike momentum/reversal, a raw 60-day window
would otherwise reach into the same month this factor is used to
predict.
"""

import numpy as np
import pandas as pd

PRICES_PATH = "Project-1/data/prices.csv"
OUT_PATH = "Project-1/data/volatility.csv"
WINDOW = 60
TRADING_DAYS_PER_YEAR = 252

if __name__ == "__main__":
    prices = pd.read_csv(PRICES_PATH, index_col=0, parse_dates=True)

    daily_returns = prices.pct_change(fill_method=None)  # never forward-fill a gap into a fake zero-vol stretch
    rolling_vol = daily_returns.rolling(WINDOW).std() * np.sqrt(TRADING_DAYS_PER_YEAR)

    monthly_vol = rolling_vol.resample("ME").last()
    volatility = monthly_vol.shift(1)
    volatility.to_csv(OUT_PATH)

    non_null_counts = volatility.notna().sum(axis=1)
    print("Stocks with a valid volatility value, by month (first 10 non-empty months):")
    print(non_null_counts[non_null_counts > 0].head(10))

    print("\nCross-sectional (annualised) volatility stats for the most recent month:")
    print(volatility.iloc[-1].describe())
