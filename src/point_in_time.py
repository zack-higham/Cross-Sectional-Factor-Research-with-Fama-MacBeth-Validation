"""
Robustness extension 3: point-in-time S&P 500 membership. Specification
fixed in robustness_spec.py before this was run.

The main universe is the 200 largest S&P 500 members in September 2026,
applied to the whole history. Here the universe in month t is restricted
to those same 200 stocks *only in months when they were index members at
the end of month t-1* (the last dated constituent list on or before that
month-end). A stock that joined the index in 2020 therefore drops out of
2012 to 2020, removing the main channel of end-of-sample selection: stocks
present early only because they later grew into the 2026 top 200.

What it cannot fix: firms that left the index (acquired, shrank, failed)
are still missing because their prices are not in the panel, so the
remaining survivorship bias is smaller but not zero.

Everything else is identical to the main analysis: the same factor values
(computed from the same prices), the 1st/99th percentile winsorisation
re-applied each month to the membership-filtered cross-section, the same
deciles, 100-stock minimum and Fama-MacBeth specification.

Ticker mapping. The membership file records the symbol in use on each
date, so renamed companies appear under their old symbol before the
rename. Each alias below was identified by auditing every panel stock
whose membership began after its price history did (old symbol leaving
the index on the same day the new one entered) and checked against the
corporate event. Aliases are date-bounded: IR, for example, belonged to
Ingersoll-Rand plc (now TT) only until 2020-03-03, after which the symbol
passed to a different company. Recycled symbols whose earlier owner left
the index before the panel stock's prices begin (CEG, DELL, SNDK) are
harmless, because a stock with no price has no factor value; membership
is nevertheless only counted on dates with a price.
"""

import glob

import numpy as np
import pandas as pd

from decile_backtest import PRICES_PATH, monthly_returns
from extensions import decile_backtest
from fama_macbeth import cross_sectional_regressions, time_series_test
from risk_adjustment import capm, market_excess
from robustness_spec import (SIZE_PARTIAL_SHRINK, SIZE_SUPPORT_SHRINK,
                             SP500_HISTORY_COMMIT, T_THRESHOLD)
from validation_costs import spread_stats

OUT_DIR = "output"
FACTORS = ["momentum", "reversal", "volatility", "size"]
RAW_FILES = {
    "momentum": "data/momentum.csv",
    "reversal": "data/reversal.csv",
    "volatility": "data/volatility.csv",
    "size": "data/size.csv",
    "beta": "data/beta.csv",
}
WINSORISED = ["momentum", "reversal", "volatility", "beta"]  # size is never winsorised

# panel ticker -> [(historical symbol, valid until (exclusive) or None)]
ALIASES = {
    "COR": [("ABC", "2023-08-30")],    # AmerisourceBergen renamed Cencora
    "BNY": [("BK", "2026-05-21")],     # Bank of New York Mellon symbol change
    "ELV": [("ANTM", "2022-06-28")],   # Anthem renamed Elevance Health
    "HWM": [("ARNC", "2020-04-06")],   # Arconic Inc renamed Howmet after the 2020 split
    "META": [("FB", "2022-06-09")],    # Facebook renamed Meta Platforms
    "MRSH": [("MMC", "2026-01-14")],   # Marsh McLennan symbol change
    "RTX": [("UTX", "2020-04-03")],    # United Technologies renamed RTX after the Raytheon merger
    "TFC": [("BBT", "2019-12-09")],    # BB&T renamed Truist after the SunTrust merger
    "TT": [("IR", "2020-03-03")],      # Ingersoll-Rand plc renamed Trane; IR then reused
    "WBD": [("DISCA", "2022-04-11")],  # Discovery renamed Warner Bros. Discovery
    "LIN": [("PX", "2018-11-06")],     # Praxair succeeded by Linde plc (see BRIDGES)
    "GOOG": [("GOOGL", None)],         # class C of Alphabet; file lists class A from 2010
}
# Linde plc succeeded Praxair in the index on the October 2018 merger, but the
# file has a six-day gap (PX last listed 2018-10-30, LIN first 2018-11-06).
BRIDGES = {"LIN": ("2018-10-31", "2018-11-05")}


def load_history():
    path = sorted(glob.glob(f"data/raw/*sp500_history_{SP500_HISTORY_COMMIT[:7]}.csv"))[-1]
    h = pd.read_csv(path)
    h["date"] = pd.to_datetime(h["date"])
    return h.sort_values("date").set_index("date")["tickers"].str.split(",").apply(set)


def daily_membership(history, tickers):
    """Boolean frame (list dates x panel tickers), applying aliases and bridges."""
    out = {}
    for t in tickers:
        own = t.replace("-", ".")  # BRK-B (Yahoo) is BRK.B in the file
        m = history.apply(lambda s: own in s)
        for sym, until in ALIASES.get(t, []):
            alias = history.apply(lambda s: sym in s)
            if until is not None:
                alias &= history.index < pd.Timestamp(until)
            m |= alias
        if t in BRIDGES:
            a, b = BRIDGES[t]
            m |= (history.index >= pd.Timestamp(a)) & (history.index <= pd.Timestamp(b))
        out[t] = m
    return pd.DataFrame(out)


def month_end_membership(daily, month_ends):
    """Membership on each month-end = last dated list on or before it."""
    return daily.astype(float).reindex(daily.index.union(month_ends)).ffill().loc[month_ends].astype(bool)


def winsorise(df, lower=0.01, upper=0.99):
    lo, hi = df.quantile(lower, axis=1), df.quantile(upper, axis=1)
    return df.clip(lower=lo, upper=hi, axis=0)


def summary_row(ls_stats, fm4, fmb, alpha4, alphab, f):
    return {
        "ls_ann_ret_%": ls_stats.loc[f, "ann_ret_%"], "ls_sharpe": ls_stats.loc[f, "sharpe"],
        "ls_nw_t": ls_stats.loc[f, "nw_t"], "ls_months": ls_stats.loc[f, "months"],
        "fm4_mean_%": fm4.loc[f, "mean_%"], "fm4_nw_t": fm4.loc[f, "nw_t"],
        "fmb_mean_%": fmb.loc[f, "mean_%"], "fmb_nw_t": fmb.loc[f, "nw_t"],
        "fm4_mktadj_alpha_%": alpha4.loc[f, "alpha_ann_%"], "fm4_mktadj_t": alpha4.loc[f, "alpha_nw_t"],
        "fmb_mktadj_alpha_%": alphab.loc[f, "alpha_ann_%"], "fmb_mktadj_t": alphab.loc[f, "alpha_nw_t"],
    }


if __name__ == "__main__":
    prices = pd.read_csv(PRICES_PATH, index_col=0, parse_dates=True)
    returns = monthly_returns(prices)
    raw = {k: pd.read_csv(p, index_col=0, parse_dates=True) for k, p in RAW_FILES.items()}

    history = load_history()
    daily = daily_membership(history, prices.columns)
    month_ends = raw["momentum"].index
    member_end = month_end_membership(daily, month_ends)
    has_price = prices.resample("ME").last().reindex(month_ends).notna()
    member_end &= has_price
    # factor row t is known at the end of t-1, so membership is taken at the end of t-1
    member = member_end.shift(1, fill_value=False).astype(bool)

    pit = {}
    for k, df in raw.items():
        masked = df.where(member)
        pit[k] = winsorise(masked) if k in WINSORISED else masked
    factors = {k: pit[k] for k in FACTORS}

    # sample sizes
    ret_months = returns.index
    n_members = member.loc[ret_months].sum(axis=1)
    common_main = pd.DataFrame({k: pd.read_csv(RAW_FILES[k], index_col=0, parse_dates=True).stack()
                                for k in FACTORS}).dropna()
    common = pd.concat([factors[k].stack().rename(k) for k in FACTORS], axis=1).dropna()
    has_ret = returns.stack().rename("ret")
    n_common_pit = common.join(has_ret, how="inner").groupby(level=0).size()
    n_common_main = common_main.join(has_ret, how="inner").groupby(level=0).size()
    counts = pd.DataFrame({"pit_members": n_members, "pit_common": n_common_pit,
                           "main_common": n_common_main}).loc["2012-02":].fillna(0).astype(int)
    counts.to_csv(f"{OUT_DIR}/pit_counts.csv")

    # decile spreads
    ls = pd.DataFrame({k: decile_backtest(factors[k], returns)[0] for k in FACTORS})
    ls.to_csv(f"{OUT_DIR}/pit_long_short_returns.csv")
    ls_stats = spread_stats(ls)

    # Fama-MacBeth: four-factor and with beta control, plus market-adjusted slope alphas
    slopes4, r2_4, n4 = cross_sectional_regressions(returns, factors)
    slopesb, _, nb = cross_sectional_regressions(returns, {**factors, "beta": pit["beta"]})
    slopes4.to_csv(f"{OUT_DIR}/pit_fm_slopes_multivariate.csv")
    slopesb.to_csv(f"{OUT_DIR}/pit_fm_slopes_with_beta.csv")
    fm4, fmb = time_series_test(slopes4), time_series_test(slopesb)
    alpha4 = capm(slopes4[FACTORS], market_excess(slopes4.index)[0])
    alphab = capm(slopesb[FACTORS + ["beta"]], market_excess(slopesb.index)[0])

    # main-analysis comparators from the existing outputs
    main_ls = pd.read_csv(f"{OUT_DIR}/long_short_returns.csv", index_col=0, parse_dates=True)
    m_slopes4 = pd.read_csv(f"{OUT_DIR}/fm_slopes_multivariate.csv", index_col=0, parse_dates=True)
    m_slopesb = pd.read_csv(f"{OUT_DIR}/fm_slopes_with_beta.csv", index_col=0, parse_dates=True)
    main = {f: summary_row(spread_stats(main_ls), time_series_test(m_slopes4), time_series_test(m_slopesb),
                           capm(m_slopes4[FACTORS], market_excess(m_slopes4.index)[0]),
                           capm(m_slopesb[FACTORS + ["beta"]], market_excess(m_slopesb.index)[0]), f)
            for f in FACTORS}
    pitr = {f: summary_row(ls_stats, fm4, fmb, alpha4, alphab, f) for f in FACTORS}
    comparison = pd.concat({"main": pd.DataFrame(main).T, "pit": pd.DataFrame(pitr).T}, names=["universe", "factor"])
    comparison.to_csv(f"{OUT_DIR}/pit_comparison.csv")

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(f"Membership file commit {SP500_HISTORY_COMMIT[:7]}, last list {history.index.max():%Y-%m-%d}")
    yearly = counts.groupby(counts.index.year).mean().round(0)
    print("\nAverage stocks per month by year (members of panel; complete-factor sample, PIT vs main):")
    print(yearly.T.to_string())
    print(f"\nFM four-factor months: PIT {len(slopes4)} ({slopes4.index.min():%Y-%m} to {slopes4.index.max():%Y-%m}), "
          f"avg {n4.mean():.0f} stocks; main {len(m_slopes4)}. Months lost: {len(m_slopes4) - len(slopes4)}")
    print(f"Months with <100 complete-factor stocks under PIT: {(counts['pit_common'] < 100).sum()}")
    print("\nMain vs point-in-time:")
    print(comparison.astype(float).round(2).T)

    # pre-registered interpretation rule
    m_size, p_size = comparison.loc[("main", "size")], comparison.loc[("pit", "size")]
    shrink = 1 - abs(p_size["ls_ann_ret_%"]) / abs(m_size["ls_ann_ret_%"])
    if shrink >= SIZE_SUPPORT_SHRINK or abs(p_size["fm4_nw_t"]) < T_THRESHOLD:
        verdict = "SUPPORTED"
    elif shrink >= SIZE_PARTIAL_SHRINK:
        verdict = "PARTIALLY SUPPORTED"
    else:
        verdict = "NOT CONFIRMED"
    vol_t = comparison.loc[("pit", "volatility"), "fmb_mktadj_t"]
    mom_t = comparison.loc[("pit", "momentum"), "fm4_mktadj_t"]
    print(f"\nRule: size L/S magnitude shrinks {shrink:.1%}, FM size t {p_size['fm4_nw_t']:.2f} -> {verdict}")
    print(f"Rule: residual-vol (beta control, market-adjusted) t {vol_t:.2f} -> "
          f"{'supported' if vol_t < T_THRESHOLD else 'not confirmed'}")
    print(f"Rule: momentum market-adjusted four-factor t {mom_t:.2f} -> "
          f"{'robust' if mom_t > T_THRESHOLD else 'not robust'}")
    pd.Series({"size_shrink": shrink, "size_verdict": verdict, "vol_resid_t": vol_t,
               "mom_mktadj_t": mom_t}).to_csv(f"{OUT_DIR}/pit_verdicts.csv")
