"""
Robustness extension 2 (data): monthly Fama-French five factors (2x3)
and the momentum factor (UMD) from the Kenneth French Data Library.

The raw zip files are kept in data/raw/ with the download date, as an
audit trail. The CSVs inside are in percent and hold a monthly block
followed by an annual block; only the monthly block (YYYYMM dates) is
parsed. Output is one monthly file in decimal returns, indexed by
calendar month-end so it lines up with the rest of the project.
"""

import io
import os
import zipfile
from datetime import date

import pandas as pd
import requests

from robustness_spec import FRENCH_BASE_URL, FRENCH_FILES

RAW_DIR = "data/raw"
OUT_PATH = "data/french_factors.csv"


def monthly_block(raw_csv):
    """Parse the monthly section (rows dated YYYYMM) of a French CSV."""
    lines = raw_csv.splitlines()
    header = next(i for i, l in enumerate(lines) if l.startswith(","))  # first column-name row
    rows = []
    for line in lines[header + 1:]:
        first = line.split(",")[0].strip()
        if not (first.isdigit() and len(first) == 6):
            break  # monthly block ends where the annual block (YYYY) begins
        rows.append(line)
    cols = ["date"] + [c.strip() for c in lines[header].split(",")[1:]]
    df = pd.read_csv(io.StringIO("\n".join(rows)), header=None, names=cols)
    df.index = pd.to_datetime(df.pop("date").astype(str), format="%Y%m") + pd.offsets.MonthEnd(0)
    return df.astype(float) / 100


if __name__ == "__main__":
    os.makedirs(RAW_DIR, exist_ok=True)
    stamp = date.today().isoformat()
    frames = {}
    for key, fname in FRENCH_FILES.items():
        resp = requests.get(FRENCH_BASE_URL + fname, timeout=60)
        resp.raise_for_status()
        with open(f"{RAW_DIR}/{stamp}_{fname}", "wb") as fh:
            fh.write(resp.content)
        with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
            frames[key] = monthly_block(z.read(z.namelist()[0]).decode("latin-1"))

    mom = frames["mom"].rename(columns={"Mom": "UMD"})
    factors = frames["ff5"].join(mom, how="inner")
    factors.to_csv(OUT_PATH)
    print(f"Downloaded {stamp}; {len(factors)} months, "
          f"{factors.index.min():%Y-%m} to {factors.index.max():%Y-%m}")
    print(factors.loc["2011":].describe().T.round(4))
