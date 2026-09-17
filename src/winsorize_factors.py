"""
Stage 2e (part 2): winsorize momentum, reversal, and volatility
cross-sectionally, one month at a time, capping at the 1st/99th
percentile. Two-sided on purpose: catches values that are artificially
too high (e.g. SNDK during a sector rally) and ones that are
artificially too low (e.g. VRT's pre-merger SPAC-shell period, where
near-zero volatility reflects a dormant shell, not real trading).

Size is excluded - the log transform already fixed its scale problem,
confirmed by the full-panel check showing no outliers there.
"""

import pandas as pd

FACTOR_FILES = {
    "momentum": "Project-1/data/momentum.csv",
    "reversal": "Project-1/data/reversal.csv",
    "volatility": "Project-1/data/volatility.csv",
}
LOWER_PCT = 0.01
UPPER_PCT = 0.99

if __name__ == "__main__":
    for name, path in FACTOR_FILES.items():
        df = pd.read_csv(path, index_col=0, parse_dates=True)

        lower = df.quantile(LOWER_PCT, axis=1)
        upper = df.quantile(UPPER_PCT, axis=1)
        winsorized = df.clip(lower=lower, upper=upper, axis=0)

        out_path = path.replace(".csv", "_winsorized.csv")
        winsorized.to_csv(out_path)

        n_clipped = (df.lt(lower, axis=0) | df.gt(upper, axis=0)).sum().sum()
        n_valid = df.notna().sum().sum()
        print(f"{name}: clipped {n_clipped} of {n_valid} stock-month values ({n_clipped / n_valid:.2%}) -> {out_path}")
