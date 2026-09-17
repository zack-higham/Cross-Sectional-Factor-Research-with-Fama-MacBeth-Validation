"""
Stage 2e (part 1): full-panel sanity check on factor distributions.
Earlier per-script printouts only checked the most recent month; this
looks across every month for all four factors, to see the real shape
of the data - and whether outliers are isolated incidents or a
recurring pattern - before deciding how aggressively to winsorize.
"""

import pandas as pd

FACTOR_FILES = {
    "momentum": "Project-1/data/momentum.csv",
    "reversal": "Project-1/data/reversal.csv",
    "volatility": "Project-1/data/volatility.csv",
    "size": "Project-1/data/size.csv",
}

if __name__ == "__main__":
    for name, path in FACTOR_FILES.items():
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        stacked = df.stack()

        print(f"\n=== {name} ===")
        print(f"Full-panel distribution ({len(stacked)} stock-month observations):")
        print(stacked.describe())

        print("\nTop 10 highest values (date, ticker -> value):")
        print(stacked.sort_values(ascending=False).head(10))

        print("\nTop 10 lowest values (date, ticker -> value):")
        print(stacked.sort_values(ascending=True).head(10))
