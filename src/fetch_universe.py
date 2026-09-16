"""
Stage 1a: pull the current S&P 500 constituent list (ticker + GICS sector)
from Wikipedia, clean it for yfinance, and cache it to CSV.
"""

from io import StringIO

import pandas as pd
import requests

WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
OUT_PATH = "project1_fama_macbeth/data/sp500_constituents.csv"

# Wikipedia rejects requests without a browser-like User-Agent (returns 403)
HEADERS = {"User-Agent": "Mozilla/5.0 (research script; pandas read_html)"}


def fetch_constituents() -> pd.DataFrame:
    response = requests.get(WIKI_URL, headers=HEADERS)
    response.raise_for_status()

    tables = pd.read_html(StringIO(response.text))
    df = tables[0]  # first table on the page is the current constituent list

    df = df.rename(columns={
        "Symbol": "ticker",
        "Security": "name",
        "GICS Sector": "sector",
        "GICS Sub-Industry": "sub_industry",
    })[["ticker", "name", "sector", "sub_industry"]]

    # yfinance uses '-' where Wikipedia uses '.' for share classes, e.g. BRK.B -> BRK-B
    df["ticker"] = df["ticker"].str.replace(".", "-", regex=False)

    return df


if __name__ == "__main__":
    df = fetch_constituents()
    df.to_csv(OUT_PATH, index=False)

    print(f"Fetched {len(df)} constituents")
    print(df.head())
    print("\nSector counts:")
    print(df["sector"].value_counts())
