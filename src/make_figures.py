"""
Stage 7b: figures for the paper, built only from the saved output CSVs
(so every figure matches the numbers in the tables). Saved as vector
PDFs into the paper's figures folder for Overleaf.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT_DIR = "Project-1/output"
FIG_DIR = "Project-1/paper/figures"
FACTORS = ["momentum", "reversal", "volatility", "size"]
LABELS = {"momentum": "Momentum (12-1)", "reversal": "Reversal (1m)",
          "volatility": "Volatility (60d)", "size": "Size (log cap)"}
# categorical slots in fixed order (validated palette), one per factor
COLORS = dict(zip(FACTORS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]))
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"
REGIME_COLORS = ["#9ec5f4", "#3987e5", "#184f95"]  # one-hue ordinal ramp: low, mid, high VIX
SPLIT_DATE = pd.Timestamp("2020-10-31")  # last in-sample month (Stage 5)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5, "axes.titlesize": 9,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False,
    "lines.linewidth": 1.4, "savefig.bbox": "tight", "savefig.dpi": 300,
})


def load(name):
    return pd.read_csv(f"{OUT_DIR}/{name}", index_col=0, parse_dates=True)


def mark_split(ax):
    ax.axvline(SPLIT_DATE, color=MUTED, lw=0.8, ls="--")


def cumulative_long_short():
    ls = load("long_short_returns.csv")
    fig, ax = plt.subplots(figsize=(6.3, 3.2))
    for f in FACTORS:
        wealth = (1 + ls[f].dropna()).cumprod()
        ax.plot(wealth.index, wealth, color=COLORS[f], label=LABELS[f])
        ax.annotate(LABELS[f].split(" ")[0], (wealth.index[-1], wealth.iloc[-1]),
                    xytext=(4, 0), textcoords="offset points", va="center", color=INK, fontsize=7.5)
    ax.set_yscale("log")
    ax.axhline(1, color=MUTED, lw=0.8)
    mark_split(ax)
    ax.text(SPLIT_DATE, ax.get_ylim()[1], " out-of-sample", va="top", color=MUTED, fontsize=7.5)
    ax.set_ylabel("Growth of $1 (log scale)")
    ax.legend(loc="upper left", ncol=2, fontsize=7.5)
    fig.savefig(f"{FIG_DIR}/cumulative_long_short.pdf")


def decile_bars():
    dec = pd.read_csv(f"{OUT_DIR}/decile_returns.csv", index_col=0, header=[0, 1], parse_dates=True)
    fig, axes = plt.subplots(2, 2, figsize=(6.3, 4.2), sharey=True)
    for ax, f in zip(axes.flat, FACTORS):
        means = dec[f].mean() * 100
        ax.bar(range(1, 11), means.values, color=COLORS[f], width=0.8, edgecolor="white", linewidth=1)
        ax.set_title(LABELS[f], loc="left")
        ax.set_xticks(range(1, 11), [f"D{d}" for d in range(1, 11)], fontsize=7)
        ax.grid(axis="x", visible=False)
    for ax in axes[:, 0]:
        ax.set_ylabel("Mean monthly return (%)")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/decile_returns.pdf")


def rolling_panels(series, filename, ylabel, scale=1.0, window=12):
    """Monthly values (faint) + rolling mean with a 95% band, per factor."""
    fig, axes = plt.subplots(2, 2, figsize=(6.3, 4.4), sharex=True)
    for ax, f in zip(axes.flat, FACTORS):
        s = series[f].dropna() * scale
        roll_mean = s.rolling(window).mean()
        roll_se = s.rolling(window).std() / np.sqrt(window)
        ax.plot(s.index, s, color=COLORS[f], lw=0.5, alpha=0.35)
        ax.fill_between(s.index, roll_mean - 1.96 * roll_se, roll_mean + 1.96 * roll_se,
                        color=COLORS[f], alpha=0.2, linewidth=0)
        ax.plot(roll_mean.index, roll_mean, color=COLORS[f], lw=1.4)
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.axhline(s.mean(), color=INK, lw=0.8, ls=":")
        mark_split(ax)
        ax.set_title(LABELS[f], loc="left")
        ax.xaxis.set_major_locator(mdates.YearLocator(4))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    for ax in axes[:, 0]:
        ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/{filename}")


def sector_neutral_bars():
    cmp = pd.read_csv(f"{OUT_DIR}/sector_neutral_comparison.csv", index_col=0)
    x = np.arange(len(FACTORS))
    fig, ax = plt.subplots(figsize=(6.3, 2.8))
    bars = [("raw_sharpe", "Raw (full universe ranking)", "#2a78d6", -0.2),
            ("sn_sharpe", "Sector-neutral (within-sector ranking)", "#eb6834", 0.2)]
    for col, label, color, offset in bars:
        vals = cmp.loc[FACTORS, col]
        ax.bar(x + offset, vals, width=0.38, color=color, label=label, edgecolor="white", linewidth=1)
        for xi, v in zip(x + offset, vals):
            ax.annotate(f"{v:.2f}", (xi, v), xytext=(0, 3 if v >= 0 else -9),
                        textcoords="offset points", ha="center", fontsize=7.5, color=INK)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xticks(x, [LABELS[f] for f in FACTORS])
    ax.grid(axis="x", visible=False)
    ax.set_ylim(-1.7, 1.05)  # headroom so value labels clear the axes
    ax.set_ylabel("Annualised Sharpe (gross)")
    ax.legend(loc="lower left", fontsize=7.5)
    fig.savefig(f"{FIG_DIR}/sector_neutral_comparison.pdf")


def regime_bars():
    reg = pd.read_csv(f"{OUT_DIR}/regime_long_short.csv", index_col=0)
    x = np.arange(len(FACTORS))
    fig, ax = plt.subplots(figsize=(6.3, 2.8))
    for i, (g, color) in enumerate(zip(["low", "mid", "high"], REGIME_COLORS)):
        ax.bar(x + (i - 1) * 0.26, reg.loc[FACTORS, f"{g}_mean_%"], width=0.24, color=color,
               label=f"{g.capitalize()} VIX tercile", edgecolor="white", linewidth=1)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xticks(x, [LABELS[f] for f in FACTORS])
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("Mean monthly L/S return (%)")
    ax.legend(loc="lower left", fontsize=7.5)
    fig.savefig(f"{FIG_DIR}/vix_regimes.pdf")


if __name__ == "__main__":
    cumulative_long_short()
    decile_bars()
    rolling_panels(load("fm_slopes_multivariate.csv"), "fm_coefficients.pdf", "FM slope (%)", scale=100)
    rolling_panels(load("ic_monthly.csv"), "ic_timeseries.pdf", "Rank IC")
    sector_neutral_bars()
    regime_bars()
    print("figures written to", FIG_DIR)
