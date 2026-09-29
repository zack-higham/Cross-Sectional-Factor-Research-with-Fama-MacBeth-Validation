"""
Stage 7c: LaTeX tables for the paper, generated from the saved output
CSVs so that no number in the paper is typed by hand. Each file holds
only a booktabs `tabular`; captions and labels live in main.tex.
"""

import numpy as np
import pandas as pd

from validation_costs import spread_stats

OUT_DIR = "Project-1/output"
DATA_DIR = "Project-1/data"
TAB_DIR = "Project-1/paper/tables"
FACTORS = ["momentum", "reversal", "volatility", "size"]
NAMES = {"momentum": "Momentum", "reversal": "Reversal", "volatility": "Volatility", "size": "Size"}


def num(x, dp=2):
    """Math-mode number so minus signs typeset correctly."""
    if pd.isna(x):
        return "n/a"
    return f"${x:.{dp}f}$"


def tstat(x):
    return f"$({x:.2f})$"


def load(name):
    return pd.read_csv(f"{OUT_DIR}/{name}", index_col=0, parse_dates=False)


def write(name, header, rows, colspec):
    lines = [f"\\begin{{tabular}}{{{colspec}}}", "\\toprule", header + " \\\\", "\\midrule"]
    lines += [r + " \\\\" if r not in ("\\midrule",) else r for r in rows]
    lines += ["\\bottomrule", "\\end{tabular}"]
    with open(f"{TAB_DIR}/{name}.tex", "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def factor_summary():
    # only months that have a return to predict, so counts match the other tables
    last_month = pd.read_csv(f"{OUT_DIR}/long_short_returns.csv", index_col=0, parse_dates=True).index.max()
    rows = []
    for f in FACTORS:
        path = f"{DATA_DIR}/{f}.csv" if f == "size" else f"{DATA_DIR}/{f}_winsorized.csv"
        panel = pd.read_csv(path, index_col=0, parse_dates=True)
        valid = panel.dropna(how="all").loc[:last_month]
        pooled = valid.stack()
        rows.append(" & ".join([NAMES[f], f"{valid.index.min():%Y-%m}", str(len(valid)),
                                f"${valid.notna().sum(axis=1).mean():.0f}$",
                                num(pooled.mean(), 3), num(pooled.std(), 3),
                                num(pooled.quantile(0.01), 3), num(pooled.quantile(0.99), 3)]))
    write("factor_summary", "Factor & First month & Months & Avg.\\ stocks & Mean & Std.\\ dev. & P1 & P99",
          rows, "lrrrrrrr")


def decile_spreads():
    ls = load("long_short_returns.csv")
    to = load("turnover.csv")
    st = spread_stats(ls)
    rows = [" & ".join([NAMES[f], str(int(st.loc[f, "months"])), num(ls[f].mean() * 100),
                        num(st.loc[f, "ann_ret_%"]), num(st.loc[f, "ann_vol_%"]),
                        num(st.loc[f, "sharpe"]), num(st.loc[f, "nw_t"]), num(to[f].mean())])
            for f in FACTORS]
    write("decile_spreads", "Factor & Months & Mean (\\%/mo) & Ann.\\ return (\\%) & Ann.\\ vol (\\%) "
          "& Sharpe & NW $t$ & Turnover", rows, "lrrrrrrr")


def decile_means():
    dec = pd.read_csv(f"{OUT_DIR}/decile_returns.csv", index_col=0, header=[0, 1])
    rows = [" & ".join([NAMES[f]] + [num(dec[f][str(d)].mean() * 100) for d in range(1, 11)])
            for f in FACTORS]
    write("decile_means", "Factor & " + " & ".join(f"D{d}" for d in range(1, 11)), rows, "l" + "r" * 10)


def fama_macbeth():
    uni = load("fm_summary_univariate.csv")
    multi = load("fm_summary_multivariate.csv")
    rows = []
    for f in FACTORS + ["const"]:
        u = [num(uni.loc[f, "mean_%"], 3), tstat(uni.loc[f, "nw_t"])] if f in uni.index else ["", ""]
        m = multi.loc[f]
        label = "Intercept" if f == "const" else NAMES[f]
        rows.append(" & ".join([label] + u + [num(m["mean_%"], 3), tstat(m["nw_t"]),
                                              tstat(m["fm_t"]), num(m["pct_months_pos"], 1)]))
    write("fama_macbeth", "& \\multicolumn{2}{c}{Univariate} & \\multicolumn{4}{c}{Multivariate} \\\\\n"
          "\\cmidrule(lr){2-3} \\cmidrule(lr){4-7}\n"
          "Factor & $\\bar\\gamma$ (\\%) & NW $t$ & $\\bar\\gamma$ (\\%) & NW $t$ & Plain FM $t$ & \\% months $>0$",
          rows, "lrrrrrr")


def is_oos():
    d = pd.read_csv(f"{OUT_DIR}/is_oos_comparison.csv", index_col=[0, 1])
    rows = []
    for f in FACTORS:
        i, o = d.loc[("in_sample", f)], d.loc[("out_of_sample", f)]
        rows.append(" & ".join([NAMES[f], num(i["ls_ann_ret_%"]), num(i["ls_sharpe"]), num(i["fm_mean_%"], 3),
                                tstat(i["fm_nw_t"]), num(o["ls_ann_ret_%"]), num(o["ls_sharpe"]),
                                num(o["fm_mean_%"], 3), tstat(o["fm_nw_t"])]))
    dates = pd.read_csv(f"{OUT_DIR}/fm_slopes_multivariate.csv", index_col=0, parse_dates=True).index
    split = dates[int(len(dates) * 0.6) - 1]  # same rule as validation_costs.py
    span = lambda idx: f"{idx.min():%Y-%m} to {idx.max():%Y-%m}, {len(idx)} months"
    write("is_oos", f"& \\multicolumn{{4}}{{c}}{{In-sample ({span(dates[dates <= split])})}} & "
          f"\\multicolumn{{4}}{{c}}{{Out-of-sample ({span(dates[dates > split])})}} \\\\\n"
          "\\cmidrule(lr){2-5} \\cmidrule(lr){6-9}\n"
          "Factor & L/S ret.\\ (\\%) & Sharpe & FM $\\bar\\gamma$ (\\%) & NW $t$ "
          "& L/S ret.\\ (\\%) & Sharpe & FM $\\bar\\gamma$ (\\%) & NW $t$", rows, "lrrrrrrrr")


def costs():
    c = load("costs_summary.csv")
    s = load("cost_sensitivity_sharpe.csv")
    rows = [" & ".join([NAMES[f], num(c.loc[f, "mean_turnover"]), num(c.loc[f, "ann_cost_%"]),
                        num(c.loc[f, "gross_ann_ret_%"]), num(c.loc[f, "net_ann_ret_%"]),
                        num(c.loc[f, "breakeven_bps"], 1)] + [num(s.loc[f, col]) for col in s.columns])
            for f in FACTORS]
    write("costs", "& & & & & & \\multicolumn{4}{c}{Net Sharpe at cost (bps per unit turnover)} \\\\\n"
          "\\cmidrule(lr){7-10}\n"
          "Factor & Turnover & Cost (\\%/yr) & Gross ret.\\ (\\%) & Net ret.\\ (\\%) & Break-even (bps) "
          "& 0 & 5 & 10 & 20", rows, "lrrrrrrrrr")


def sector_neutral():
    c = load("sector_neutral_comparison.csv")
    fm = pd.read_csv(f"{OUT_DIR}/fm_sector_neutral_comparison.csv", index_col=0, header=[0, 1])
    rows = [" & ".join([NAMES[f], num(c.loc[f, "raw_ann_ret_%"]), num(c.loc[f, "sn_ann_ret_%"]),
                        num(c.loc[f, "raw_sharpe"]), num(c.loc[f, "sn_sharpe"]), num(c.loc[f, "sn_net_sharpe"]),
                        num(c.loc[f, "corr_raw_sn"]),
                        num(fm.loc[f, ("raw", "mean_%")], 3) + " " + tstat(fm.loc[f, ("raw", "nw_t")]),
                        num(fm.loc[f, ("sector_neutral", "mean_%")], 3) + " " + tstat(fm.loc[f, ("sector_neutral", "nw_t")])])
            for f in FACTORS]
    write("sector_neutral", "& \\multicolumn{2}{c}{Ann.\\ return (\\%)} & \\multicolumn{3}{c}{Sharpe} & & "
          "\\multicolumn{2}{c}{FM $\\bar\\gamma$ (\\%), NW $t$} \\\\\n"
          "\\cmidrule(lr){2-3} \\cmidrule(lr){4-6} \\cmidrule(lr){8-9}\n"
          "Factor & Raw & SN & Raw & SN & SN net & Corr. & Raw & Sector dummies", rows, "lrrrrrrrr")


def regimes():
    ls = load("regime_long_short.csv")
    fm = load("regime_fm_slopes.csv")
    rows = [" & ".join([NAMES[f], num(ls.loc[f, "low_mean_%"]), num(ls.loc[f, "mid_mean_%"]),
                        num(ls.loc[f, "high_mean_%"]), num(ls.loc[f, "welch_t"]), num(ls.loc[f, "welch_p"]),
                        num(fm.loc[f, "low_mean_%"], 3), num(fm.loc[f, "high_mean_%"], 3), num(fm.loc[f, "welch_p"])])
            for f in FACTORS]
    write("regimes", "& \\multicolumn{5}{c}{Long-short return (\\%/mo)} & \\multicolumn{3}{c}{FM $\\bar\\gamma$ (\\%)} \\\\\n"
          "\\cmidrule(lr){2-6} \\cmidrule(lr){7-9}\n"
          "Factor & Low & Mid & High & Welch $t$ & $p$ & Low & High & $p$", rows, "lrrrrrrrr")


def information_coefficient():
    ic = load("ic_summary.csv")
    rows = [" & ".join([NAMES[f], str(int(ic.loc[f, "months"])), num(ic.loc[f, "mean_ic"], 3),
                        num(ic.loc[f, "ic_std"], 3), num(ic.loc[f, "icir_ann"]), num(ic.loc[f, "nw_t"]),
                        num(ic.loc[f, "pct_pos"], 1)])
            for f in FACTORS]
    write("information_coefficient", "Factor & Months & Mean IC & IC std.\\ dev. & ICIR (ann.) & NW $t$ "
          "& \\% months IC $>0$", rows, "lrrrrrr")


def subperiods():
    """Multivariate FM premia before vs during the 2023-2026 AI/semiconductor rally."""
    from fama_macbeth import time_series_test
    slopes = pd.read_csv(f"{OUT_DIR}/fm_slopes_multivariate.csv", index_col=0, parse_dates=True)
    early, late = slopes.loc[:"2022"], slopes.loc["2023":]
    a, b = time_series_test(early), time_series_test(late)
    rows = [" & ".join([NAMES[f], num(a.loc[f, "mean_%"], 3), tstat(a.loc[f, "nw_t"]),
                        num(b.loc[f, "mean_%"], 3), tstat(b.loc[f, "nw_t"])]) for f in FACTORS]
    write("subperiods", f"& \\multicolumn{{2}}{{c}}{{{early.index.min():%Y-%m} to {early.index.max():%Y-%m} "
          f"({len(early)} months)}} & \\multicolumn{{2}}{{c}}{{{late.index.min():%Y-%m} to "
          f"{late.index.max():%Y-%m} ({len(late)} months)}} \\\\\n"
          "\\cmidrule(lr){2-3} \\cmidrule(lr){4-5}\n"
          "Factor & $\\bar\\gamma$ (\\%) & NW $t$ & $\\bar\\gamma$ (\\%) & NW $t$", rows, "lrrrr")


def momentum_crashes(n=5):
    ls = pd.read_csv(f"{OUT_DIR}/long_short_returns.csv", index_col=0, parse_dates=True)
    vix = pd.read_csv(f"{DATA_DIR}/vix.csv", index_col=0, parse_dates=True)["vix"].resample("ME").last()
    worst = ls["momentum"].nsmallest(n)
    rows = [" & ".join([f"{d:%Y-%m}", num(r * 100, 1), num(vix.shift(1).loc[d], 1)]) for d, r in worst.items()]
    write("momentum_crashes", "Month & Momentum L/S return (\\%) & VIX at prior month-end", rows, "lrr")


def capm_alphas():
    a = pd.read_csv(f"{OUT_DIR}/capm_alphas.csv", index_col=[0, 1])
    cell = lambda spec, f: num(a.loc[(spec, f), "alpha_ann_%"]) + " " + tstat(a.loc[(spec, f), "alpha_nw_t"])
    rows = [" & ".join([NAMES[f], num(a.loc[("gross", f), "beta"]), tstat(a.loc[("gross", f), "beta_nw_t"]),
                        cell("gross", f), cell("net", f), cell("sector_neutral", f),
                        cell("pre_2023", f), cell("from_2023", f)]) for f in FACTORS]
    write("capm_alphas", "& \\multicolumn{2}{c}{Market beta} & \\multicolumn{5}{c}{CAPM alpha (\\%/yr), NW $t$} \\\\\n"
          "\\cmidrule(lr){2-3} \\cmidrule(lr){4-8}\n"
          "Factor & $\\beta$ & NW $t$ & Gross & Net & Sector-neutral & 2012 to 2022 & 2023 to 2026", rows, "lrrrrrrr")


def fm_risk():
    fm = load("fm_summary_with_beta.csv")
    s = pd.read_csv(f"{OUT_DIR}/fm_slope_alphas.csv", index_col=[0, 1, 2])
    cell = lambda spec, per, f: (num(s.loc[(spec, per, f), "alpha_ann_%"]) + " " + tstat(s.loc[(spec, per, f), "alpha_nw_t"])
                                 if (spec, per, f) in s.index else "")
    names = dict(NAMES, beta="Beta")
    rows = [" & ".join([names[f], num(fm.loc[f, "mean_%"], 3), tstat(fm.loc[f, "nw_t"]),
                        cell("four_factor", "full", f), cell("four_factor", "pre_2023", f),
                        cell("with_beta", "full", f), cell("with_beta", "pre_2023", f)])
            for f in FACTORS + ["beta"]]
    write("fm_risk", "& \\multicolumn{2}{c}{FM with beta control} & \\multicolumn{4}{c}{Market-adjusted slope alpha (\\%/yr), NW $t$} \\\\\n"
          "\\cmidrule(lr){2-3} \\cmidrule(lr){4-7}\n"
          "& & & \\multicolumn{2}{c}{Four-factor model} & \\multicolumn{2}{c}{With beta control} \\\\\n"
          "\\cmidrule(lr){4-5} \\cmidrule(lr){6-7}\n"
          "Factor & $\\bar\\gamma$ (\\%) & NW $t$ & Full & 2012 to 2022 & Full & 2012 to 2022", rows, "lrrrrrr")


if __name__ == "__main__":
    for fn in [factor_summary, decile_spreads, decile_means, fama_macbeth, is_oos, costs,
               sector_neutral, regimes, information_coefficient, subperiods, momentum_crashes,
               capm_alphas, fm_risk]:
        fn()
    print("tables written to", TAB_DIR)
