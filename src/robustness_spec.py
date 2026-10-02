"""
Pre-registered specification for the post-publication robustness
extensions (drawdowns, multi-factor alphas, point-in-time membership,
non-winsorised factors).

These analyses were added after the original results had been seen.
Every choice below is fixed here, and this file is committed, before
any of the new numbers is computed. None of the new analyses changes an
existing result, table or figure; they are new scripts, new output files
and new tables only. No specification below may be added, dropped or
varied after the output is seen.

1. Drawdowns (drawdowns.py)
   Portfolios: the four D10 - D1 long-short portfolios, raw and
   sector-neutral, gross and net of 10 bps per unit turnover (16 series),
   using the existing output files and the existing net_returns() rule.
   Wealth W_t = prod_{s<=t} (1 + r_s), starting from W_0 = 1 in the month
   before the first return, so a loss in the first month counts.
   Max drawdown = min_t (W_t / max_{s<=t} W_s - 1), in %, with the peak
   month, the trough month and the recovery month (first month after the
   trough with W_t >= the peak wealth), or "not recovered".
   Calmar = annualised arithmetic mean return (mean x 12, the same figure
   as the paper's "Ann. return" columns) / |max drawdown|.
   Monthly data only, so intra-month drawdowns are understated.
   Checks: momentum's MDD must be at least 31.4% (its worst month);
   size (negative mean) should have a very large, unrecovered drawdown.

2. Multi-factor alphas (fetch_french.py, multifactor_alphas.py)
   Data: Kenneth French Data Library monthly files
   F-F_Research_Data_5_Factors_2x3 and F-F_Momentum_Factor, raw zips saved
   to data/raw/ with the download date. Percent -> decimal.
   Test series: the four long-short portfolios (gross and net) and the
   four multivariate Fama-MacBeth slope series (four-factor model,
   intercept excluded).
   Timing: calendar month t on calendar month t, no lag. Alignment check:
   correlation of the paper's SPY excess return with French Mkt-RF must
   exceed 0.95, otherwise the alignment is wrong and nothing is reported.
   Models: CAPM (Mkt-RF), FF3 (+SMB, HML), Carhart (+UMD), FF5+UMD
   (Mkt-RF, SMB, HML, RMW, CMA, UMD).
   Output: annualised alpha (x12) with Newey-West t (same lag rule as the
   paper, floor(4(T/100)^(2/9)) on each regression's own T), every loading
   with NW t, R^2. Periods: full overlap, start to 2022-12, 2023-01 to end.
   If French data end before 2026-08, the overlap is used and the end
   month is stated in the table caption.

3. Point-in-time S&P 500 membership (fetch_sp500_history.py,
   point_in_time.py)
   Source: fja05680/sp500 GitHub repository, "S&P 500 Historical
   Components & Changes (Updated).csv" (dated constituent lists), pinned
   to one commit; raw file saved to data/raw/ with the download date.
   Universe in month t = stocks in the existing 200-stock price panel that
   were index members at the end of month t-1 (the last dated list on or
   before that month-end). Membership is known at the same date as the
   factor values, so no future information enters.
   Everything else identical: same factor definitions computed from the
   same prices; the 1st/99th percentile winsorisation rule re-applied each
   month to the membership-filtered cross-section (momentum, reversal,
   volatility, beta; not size); same decile rule; same 100-stock minimum
   (months below it are excluded and the number lost is reported); same
   Fama-MacBeth specification and NW lags.
   Ticker mapping: an explicit table for renames and share-class formats
   (e.g. FB -> META, BRK.B -> BRK-B). Every panel ticker that never matches
   a historical member is printed, investigated and fixed or documented.
   A historical record under a recycled ticker (a different company that
   once used the symbol) must not be credited to the panel stock; the
   membership history is cross-checked against each stock's price history.
   No prices are downloaded for former constituents.
   Report: stocks per month; decile long-short mean, Sharpe, NW t (gross);
   multivariate FM premia for the four-factor and beta-control models; and
   the market-adjusted (SPY CAPM) alphas of those FM slope series.
   Interpretation rule (fixed now):
     - Size: the survivorship diagnosis is SUPPORTED if the size long-short
       annualised return falls in magnitude by at least 50%, or the four-
       factor FM size NW |t| falls below 2; PARTIALLY supported if the
       long-short magnitude falls by 25% to 50%; otherwise NOT CONFIRMED by
       this check (the remaining bias from missing exits may still be
       large). Any of the three outcomes is reported.
     - Residual volatility: the diagnosis is supported if the market-
       adjusted volatility slope alpha in the beta-control model loses
       significance (NW t < 2); otherwise not confirmed by this check.
     - Momentum: reported as robust to this check if its market-adjusted
       four-factor slope alpha keeps NW t > 2.
   What the check does not fix: firms that left the index (acquired,
   declined, delisted) are still absent, so the remaining bias is smaller
   but not zero.

4. Non-winsorised factors (raw_factors.py)
   Momentum, reversal and volatility used without the 1st/99th percentile
   clip (size was never winsorised). Re-run: the four-factor multivariate
   Fama-MacBeth regression, its market-adjusted slope alphas, and the
   decile long-short spreads. Nothing else changes. Expected: decile sorts
   are rank-based, so they can differ only through ties created by the
   clip; the regression can differ materially, especially for momentum.
"""

# Stage 1
COST_BPS = 10
ANNUALISE = 12

# Stage 2
FRENCH_BASE_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
FRENCH_FILES = {
    "ff5": "F-F_Research_Data_5_Factors_2x3_CSV.zip",
    "mom": "F-F_Momentum_Factor_CSV.zip",
}
MODELS = {
    "CAPM": ["Mkt-RF"],
    "FF3": ["Mkt-RF", "SMB", "HML"],
    "Carhart": ["Mkt-RF", "SMB", "HML", "UMD"],
    "FF5+UMD": ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "UMD"],
}
PERIODS = {"full": (None, None), "pre_2023": (None, "2022-12"), "from_2023": ("2023-01", None)}
MIN_ALIGNMENT_CORR = 0.95

# Stage 3
# pinned to the repository's 2026-09-07 update so the input cannot change
SP500_HISTORY_COMMIT = "a2430f2af0c79ddf0748e91de11bdeb1616ab5a7"
SP500_HISTORY_URL = (f"https://raw.githubusercontent.com/fja05680/sp500/{SP500_HISTORY_COMMIT}/"
                     "S%26P%20500%20Historical%20Components%20%26%20Changes%20(Updated).csv")
SIZE_SUPPORT_SHRINK = 0.50
SIZE_PARTIAL_SHRINK = 0.25
T_THRESHOLD = 2.0

# Stage 4: raw (non-winsorised) factor files
RAW_FACTOR_FILES = {
    "momentum": "data/momentum.csv",
    "reversal": "data/reversal.csv",
    "volatility": "data/volatility.csv",
    "size": "data/size.csv",
}
