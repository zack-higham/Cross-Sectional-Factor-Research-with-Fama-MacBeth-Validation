# Cross-Sectional Factor Research with Fama-MacBeth Validation

Do momentum, short-term reversal, volatility and size predict next-month returns among the
200 largest S&P 500 stocks? I test each factor through three lenses (decile portfolio sorts,
Fama-MacBeth regressions with Newey-West standard errors, and rank information coefficients),
then stress-test the results with an out-of-sample split, market risk adjustment, transaction
costs, sector-neutral construction and a VIX regime split.

**Full paper:** [paper/Cross_Sectional_Factor_Study_with_Fama_Macbeth_Validation.pdf](paper/Cross_Sectional_Factor_Study_with_Fama_Macbeth_Validation.pdf)
(17 pages; LaTeX source in [`paper/main.tex`](paper/main.tex))

---

## Key findings

| Factor | Raw result | After market risk adjustment | Verdict |
|---|---|---|---|
| **Momentum (12-1)** | FM premium 0.30%/mo per SD, NW t = 2.91 | Close to market-neutral (beta -0.13). Market-adjusted FM premium 3.7%/yr, t = 3.41, and t = 2.49 before 2023 | The one result that survives risk adjustment |
| **Volatility (60d)** | FM premium 0.49%/mo, t = 3.07; decile Sharpe 0.77 | Long-short beta 1.17 (t = 7.0); CAPM alpha insignificant, and -3.1%/yr before 2023 | Largely market beta in a bull market |
| **Reversal (1m)** | Insignificant (t = 0.51) | Turnover 3.44/month; break-even cost 9.7 bps | Eliminated by trading costs |
| **Size (log cap)** | Big-minus-small decile spread -21%/yr; FM t = -3.85 | Negative throughout | Survivorship bias, not a size effect |

The most statistically significant raw result (volatility) turned out to be compensation for
market exposure, while momentum, the factor with the strongest prior literature, was the only
premium that survived risk adjustment. It also survives sector neutralisation, and its
break-even trading cost (56 bps per unit turnover) is well above the assumed 10 bps. Rank ICs for both are
insignificant, meaning the predictive content sits in the tails of the cross-section rather
than in a consistent ordering of all stocks. A universe selected on end-of-sample market cap
explains both the negative size premium and a residual volatility premium that contradicts
the literature.

---

## Method

**Timing convention (used everywhere).** The factor value paired with month *t*'s return uses
only prices up to the end of month *t-1*. I verified this by recomputing individual factor
values and returns by hand for sample stocks.

| Factor | Definition |
|---|---|
| Momentum | `P(t-2) / P(t-13) - 1` (months t-12 to t-2, skipping the reversal month) |
| Reversal | `P(t-1) / P(t-2) - 1` |
| Volatility | Std. dev. of the last 60 daily returns to the end of t-1, annualised |
| Size | log(market cap) at the end of t-1 |
| Beta (control) | Slope of the last 252 daily returns on SPY, to the end of t-1 |

**Three lenses.**
1. *Decile sorts:* equal-weighted D1 to D10 portfolios, D10 minus D1 long-short, turnover.
2. *Fama-MacBeth:* monthly cross-sectional OLS on z-scored factors, then time-series means of
   the slopes with Newey-West (HAC, 4-lag) t-statistics; univariate and multivariate.
3. *Information coefficient:* monthly Spearman rank correlation of factor and next-month return.

**Validation.**
- 60/40 in-sample / out-of-sample split on one common calendar date, plus a pre-2023 vs 2023+ split.
- CAPM alphas against SPY excess returns (T-bill risk-free rate); beta as a Fama-MacBeth
  control; and CAPM regressions of the monthly Fama-MacBeth slopes themselves, since a
  characteristic control does not hedge a factor-mimicking portfolio's market exposure.
- Transaction costs at 10 bps per unit turnover, with 0/5/10/20 bps sensitivity and break-even costs.
- Sector-neutral version (within-GICS-sector ranks) and sector-dummy Fama-MacBeth.
- Lagged-VIX tercile regimes (descriptive).

---

## Pipeline

Each stage was built, run and checked before the next one started; the commit history
follows the same order.

| Stage | Scripts | What it does |
|---|---|---|
| 1. Data | `fetch_universe.py`, `fetch_market_caps.py`, `fetch_prices.py`, `sanity_check_prices.py` | S&P 500 list and sectors from Wikipedia; top 200 by market cap; daily adjusted prices 2011 to 2026; checks for empty series, gaps, low coverage and >50% daily moves |
| 2. Factors | `compute_momentum.py`, `compute_reversal.py`, `compute_volatility.py`, `compute_size.py`, `check_factor_distributions.py`, `winsorize_factors.py` | Factor construction, full-panel distribution checks, per-month 1st/99th percentile winsorisation |
| 3. Deciles | `decile_backtest.py` | Decile returns, D10-D1 spreads, turnover |
| 4. Fama-MacBeth | `fama_macbeth.py` | Cross-sectional regressions with Newey-West inference |
| 5. Validation | `validation_costs.py` | In-sample vs out-of-sample, transaction costs |
| 6. Extensions | `extensions.py`, `fetch_vix.py` | Sector-neutral factors, VIX regimes |
| 6b. Risk | `fetch_market.py`, `compute_beta.py`, `risk_adjustment.py` | Betas, CAPM alphas, beta-controlled and market-adjusted Fama-MacBeth |
| 7. Reporting | `information_coefficient.py`, `make_figures.py`, `make_tables.py` | IC, and every figure and table in the paper, generated directly from the outputs |

---

## Issues found and fixed along the way

- **Momentum overlapped reversal.** My first momentum window, `P(t-1)/P(t-13)`, included month
  t-1, which is exactly the reversal window. Fixed to `P(t-2)/P(t-13)` in its own commit.
- **Partial final month.** The price data ends mid-September 2026, so the last "monthly" return
  was a half-month. A shared helper now drops an incomplete final month; all results regenerated.
- **Outliers in both tails.** SanDisk's 12-1 momentum exceeded 5,000% in 2026 (a real rally),
  while Vertiv showed near-zero volatility during its pre-merger SPAC period (a flat shell
  price). Hence two-sided winsorisation, applied per month so the cut-offs adapt to each
  volatility regime. Returns are never winsorised.
- **Hidden forward-fill.** `pct_change()` forward-fills price gaps by default, which would turn
  a gap into a fake zero-volatility stretch. Made explicit with `fill_method=None` (no ticker
  was affected).
- **Benchmark choice.** Measuring alpha against an average of the study universe gave a
  misleading answer, because that average shares the universe's survivorship bias. SPY is used
  instead.
- **Data quirks.** Wikipedia rejects the default `urllib` user agent; Yahoo uses `BRK-B` where
  Wikipedia uses `BRK.B`; three tickers failed on network timeouts and were retried rather than
  silently dropped.

---

## Limitations

- **Survivorship bias:** the universe is today's top 200, so historical small caps are
  ex-post winners. This drives the size result and is strongest near the end of the sample.
  A point-in-time constituent history (e.g. CRSP) would remove it.
- **Auxiliary look-ahead:** current share counts for market cap, current GICS labels applied
  historically, full-sample VIX tercile cut-offs. None affects return timing.
- **Risk model:** CAPM only; Fama-French multi-factor alphas are a natural extension.
- **Costs:** flat 10 bps per unit turnover; no market impact or short-borrow costs.
- **Power:** about 15 years, 20 stocks per decile, 70 out-of-sample months.

---

## Reproducing the results

```bash
pip install -r requirements.txt

# run from the repository root, in this order
python src/fetch_universe.py
python src/fetch_market_caps.py
python src/fetch_prices.py
python src/fetch_vix.py
python src/fetch_market.py
python src/sanity_check_prices.py
python src/compute_momentum.py
python src/compute_reversal.py
python src/compute_volatility.py
python src/compute_size.py
python src/compute_beta.py
python src/check_factor_distributions.py
python src/winsorize_factors.py
python src/decile_backtest.py
python src/fama_macbeth.py
python src/validation_costs.py
python src/extensions.py
python src/risk_adjustment.py
python src/information_coefficient.py
python src/make_figures.py
python src/make_tables.py
```

Downloaded data (`data/`) and intermediate results (`output/`) are not committed. They are
regenerated by the scripts. Market caps and constituents are fetched live, so a rerun on a
later date gives a slightly different universe.

## Repository structure

```
src/        pipeline scripts (one per stage)
paper/      LaTeX source, compiled PDF, generated tables/ and figures/
data/       downloaded data (not committed)
output/     intermediate results (not committed)
```

## Selected references

Fama & MacBeth (1973); Jegadeesh & Titman (1993); Carhart (1997); Ang, Hodrick, Xing & Zhang (2006);
Frazzini & Pedersen (2014); Daniel & Moskowitz (2016); Harvey, Liu & Zhu (2016); Newey & West (1987, 1994).
Full list in the paper.

---

Zack Higham · BSc Mathematics (Statistics, Financial and AI pathway), Queen Mary University of London
