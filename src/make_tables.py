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
    rows = []
    for f in FACTORS:
        path = f"{DATA_DIR}/{f}.csv" if f == "size" else f"{DATA_DIR}/{f}_winsorized.csv"
        panel = pd.read_csv(path, index_col=0, parse_dates=True)
        valid = panel.dropna(how="all")
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
    write("is_oos", "& \\multicolumn{4}{c}{In-sample (2012-02 to 2020-10, 105 months)} & "
          "\\multicolumn{4}{c}{Out-of-sample (2020-11 to 2026-09, 71 months)} \\\\\n"
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


if __name__ == "__main__":
    for fn in [factor_summary, decile_spreads, decile_means, fama_macbeth, is_oos, costs,
               sector_neutral, regimes, information_coefficient]:
        fn()
    print("tables written to", TAB_DIR)
