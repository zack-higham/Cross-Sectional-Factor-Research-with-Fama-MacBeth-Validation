"""
Stage 2d: size control - log market capitalisation. Only today's
market cap was ever fetched (Stage 1), so historical size is
approximated by holding share count constant and letting price do
the work: size(t) ~= today's market cap x (price(t) / today's price).
Log-transformed since market cap is heavily right-skewed (a $5T
mega-cap would otherwise dominate a regression on raw dollar values).
"""

import numpy as np
import pandas as pd

PRICES_PATH = "data/prices.csv"
UNIVERSE_PATH = "data/sp500_top200.csv"
OUT_PATH = "data/size.csv"

if __name__ == "__main__":
    prices = pd.read_csv(PRICES_PATH, index_col=0, parse_dates=True)
    universe = pd.read_csv(UNIVERSE_PATH).set_index("ticker")["market_cap"]

    monthly_prices = prices.resample("ME").last()
    latest_price = prices.iloc[-1]

    market_cap_proxy = monthly_prices.shift(1) * (universe / latest_price)
    size = np.log(market_cap_proxy)
    size.to_csv(OUT_PATH)

    non_null_counts = size.notna().sum(axis=1)
    print("Stocks with a valid size value, by month (first 10 non-empty months):")
    print(non_null_counts[non_null_counts > 0].head(10))

    print("\nCross-sectional log(size) stats for the most recent month:")
    print(size.iloc[-1].describe())
