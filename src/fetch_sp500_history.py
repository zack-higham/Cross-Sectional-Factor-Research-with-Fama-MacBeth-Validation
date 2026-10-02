"""
Robustness extension 3 (data): historical S&P 500 membership from the
fja05680/sp500 GitHub repository, pinned to one commit. Each row is a
date and the comma-separated list of tickers in the index from that
date. The raw file is kept in data/raw/ with the download date.
"""

import os
from datetime import date

import requests

from robustness_spec import SP500_HISTORY_COMMIT, SP500_HISTORY_URL

RAW_DIR = "data/raw"

if __name__ == "__main__":
    os.makedirs(RAW_DIR, exist_ok=True)
    resp = requests.get(SP500_HISTORY_URL, timeout=60)
    resp.raise_for_status()
    path = f"{RAW_DIR}/{date.today().isoformat()}_sp500_history_{SP500_HISTORY_COMMIT[:7]}.csv"
    with open(path, "wb") as fh:
        fh.write(resp.content)
    print(f"Saved {len(resp.content):,} bytes to {path}")
