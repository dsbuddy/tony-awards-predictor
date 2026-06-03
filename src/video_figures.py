"""
Video-ready figures for the data-storytelling video.

Distinct from make_charts.py (which makes clean README charts): these are built
for a screen-recorded video - dark background, big bold fonts, high DPI,
generous margins, one idea per frame, and a consistent palette. Each figure
maps to a beat in the script (see video_script.md). Filenames are numbered in
narration order.

    python3 video_figures.py

Output: output/video/*.png at 1920x1080-friendly sizes, 200 DPI.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

import analytics

OUT = os.path.join(os.path.dirname(__file__), "..", "output", "video")
os.makedirs(OUT, exist_ok=True)

# ---- house style (dark, bold, video-friendly) ----
BG = "#0f1117"
PANEL = "#171a21"
INK = "#f5f6fa"
MUTE = "#9aa3b2"
GOLD = "#e8b339"      # Tony / spotlight gold
GOOD = "#3fb950"
MID = "#e3a008"
BAD = "#f85149"
BLUE = "#4493f8"
GRID = "#2a2e37"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": MUTE, "ytick.color": MUTE,
    "axes.edgecolor": GRID, "grid.color": GRID,
    "font.size": 17, "font.family": "DejaVu Sans",
    "axes.titlesize": 24, "axes.titleweight": "bold",
})

FIGSIZE = (12.8, 7.2)  # 16:9


def _save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  wrote {os.path.relpath(path)}")


def _title(ax, text, sub=None):
    # place title + subtitle ABOVE the axes via the figure, so they never
    # collide with the top y-tick label
    pad = 0.10 if sub else 0.06
    ax.figure.subplots_adjust(top=1 - pad - 0.04)
    ax.text(0, 1.11 if sub else 1.05, text, transform=ax.transAxes, color=INK,
            fontsize=24, fontweight="bold", ha="left", va="bottom")
    if sub:
        ax.text(0, 1.045, sub, transform=ax.transAxes, color=MUTE,
                fontsize=15, ha="left", va="bottom")


# ---- 01: the hook - one big number ----
def fig_hook():
    o = analytics.overall()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.axis("off")
    ax.text(0.5, 0.62, f"{o['acc']*100:.0f}%", ha="center", va="center",
            fontsize=180, fontweight="bold", color=GOLD)
    ax.text(0.5, 0.34, "of Tony winners, called before the ceremony",
            ha="center", va="center", fontsize=26, color=INK)
    ax.text(0.5, 0.24, "using only the awards that came before", ha="center",
            va="center", fontsize=18, color=MUTE)
    _save(fig, "01_hook.png")


# ---- 02: the precursor timeline (the setup) ----
def fig_timeline():
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.axis("off")
    events = [("Drama Desk", 0.10), ("Outer Critics", 0.28),
              ("Drama League", 0.46), ("Critics' Circle", 0.62), ("THE TONYS", 0.90)]
    ax.plot([0.05, 0.95], [0.5, 0.5], color=GRID, lw=3, zorder=1)
    for label, x in events:
        is_tony = label == "THE TONYS"
        ax.scatter([x], [0.5], s=900 if is_tony else 420,
                   color=GOLD if is_tony else BLUE, zorder=3, edgecolor=BG, linewidth=2)
        ax.text(x, 0.62, label, ha="center", fontsize=18 if is_tony else 15,
                color=GOLD if is_tony else INK, fontweight="bold" if is_tony else "normal")
    ax.text(0.5, 0.30, "Every spring, the same shows keep winning the warm-up awards.",
            ha="center", fontsize=20, color=INK)
    ax.text(0.5, 0.21, "By Tony night, does the result already feel decided?",
            ha="center", fontsize=18, color=MUTE)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    _save(fig, "02_timeline.png")


# ---- 03: accuracy by season (does it actually work?) ----
def fig_by_year():
    data = analytics.per_year_accuracy()
    years = [str(d["year"]) for d in data]
    acc = [d["acc"] * 100 for d in data]
    base = [d["baseline_acc"] * 100 for d in data]
    o = analytics.overall()["acc"] * 100
    fig, ax = plt.subplots(figsize=FIGSIZE)
    x = np.arange(len(years))
    ax.bar(x, acc, color=GOLD, width=0.74, zorder=3, label="The model")
    ax.plot(x, base, color=BAD, lw=0, marker="_", markersize=16, mew=3,
            label="Coin-flip baseline", zorder=4)
    ax.axhline(o, color=INK, ls="--", lw=1.6, alpha=0.8)
    ax.text(len(years) - 0.4, o + 1.5, f"{o:.0f}% average", color=INK, fontsize=15, ha="right")
    ax.set_xticks(x); ax.set_xticklabels(years, rotation=45, fontsize=12)
    ax.set_ylim(0, 100); ax.set_ylabel("Winners called correctly (%)")
    ax.grid(axis="y", alpha=0.4, zorder=0)
    _title(ax, "It holds up, season after season",
           "Tested by hiding one year, training on the rest, then predicting the hidden year")
    ax.legend(loc="upper left", facecolor=PANEL, edgecolor=GRID, fontsize=14)
    _save(fig, "03_by_year.png")


# ---- 04: which categories are solved vs coin-flips ----
def fig_by_category():
    data = analytics.per_category_accuracy()
    labels = [d["label"].replace("Best ", "") for d in data]
    acc = [d["acc"] * 100 for d in data]
    colors = [GOOD if a >= 70 else (MID if a >= 55 else BAD) for a in acc]
    fig, ax = plt.subplots(figsize=(12.8, 9.5))
    y = np.arange(len(labels))
    ax.barh(y, acc, color=colors, zorder=3)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=12.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 100); ax.set_xlabel("Accuracy (%)")
    ax.axvline(50, color=MUTE, ls=":", lw=1.5)
    ax.text(50, -0.8, "coin flip", color=MUTE, fontsize=12, ha="center")
    ax.grid(axis="x", alpha=0.4, zorder=0)
    _title(ax, "Some races are basically solved. Others are a coin flip.",
           "Best Play is ~96% predictable. Sound design? Barely better than guessing.")
    _save(fig, "04_by_category.png")


# ---- 05: the calibration gut-check ----
def fig_calibration():
    bins = analytics.calibration_bins(10)
    mid = [b["predicted_mid"] * 100 for b in bins]
    act = [b["actual"] * 100 for b in bins]
    n = [b["n"] for b in bins]
    fig, ax = plt.subplots(figsize=(9.5, 9.5))
    ax.plot([0, 100], [0, 100], ls="--", color=MUTE, lw=2, label="Perfect honesty")
    ax.scatter(mid, act, s=[max(60, v * 9) for v in n], color=GOLD,
               alpha=0.85, zorder=3, edgecolor=BG, linewidth=1.5,
               label="The model (dot size = # races)")
    ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    ax.set_xlabel("When the model says this much..."); ax.set_ylabel("...this often it's right")
    ax.grid(alpha=0.4)
    _title(ax, "Does its confidence actually mean anything?",
           "On the diagonal = honest. The model's high-confidence calls land where they should.")
    ax.legend(loc="upper left", facecolor=PANEL, edgecolor=GRID, fontsize=13)
    _save(fig, "05_calibration.png")


# ---- 06: the 2026 forecast (the payoff) ----
def fig_forecast():
    data = analytics.prediction_confidences(2026)
    labels = [d["label"].replace("Best ", "") for d in data]
    probs = [d["prob"] * 100 for d in data]
    picks = [d["pick"] for d in data]
    colors = [GOOD if p > 70 else (MID if p > 45 else BAD) for p in probs]
    fig, ax = plt.subplots(figsize=(12.8, 11))
    y = np.arange(len(labels))
    ax.barh(y, probs, color=colors, zorder=3, height=0.78)
    for i, (p, pick) in enumerate(zip(probs, picks)):
        ax.text(min(p + 1, 99), i, f" {pick}", va="center", fontsize=11, color=INK)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=11.5)
    ax.invert_yaxis(); ax.set_xlim(0, 116); ax.set_xlabel("Model confidence (%)")
    ax.grid(axis="x", alpha=0.4, zorder=0)
    _title(ax, "The 2026 picks", "Green = lock | yellow = lean | red = genuine toss-up")
    _save(fig, "06_forecast_2026.png")


# ---- 07: the honest close - what it can't see ----
def fig_limits():
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.axis("off")
    ax.text(0.5, 0.84, "What the model CAN'T see", ha="center", fontsize=26,
            fontweight="bold", color=GOLD)
    lines = [
        "It reads the precursor awards, not the room.",
        "It can't feel a career-defining performance.",
        "It can't sense a vote-splitting upset brewing.",
        "Some years, the data just doesn't know either.",
    ]
    for i, t in enumerate(lines):
        ax.text(0.5, 0.62 - i * 0.13, t, ha="center", fontsize=21, color=INK)
    ax.text(0.5, 0.04, "And that's exactly why we still watch.", ha="center",
            fontsize=19, color=MUTE, style="italic")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    _save(fig, "07_limits.png")


def main():
    print(f"Rendering video figures -> {os.path.relpath(OUT)}/")
    fig_hook()
    fig_timeline()
    fig_by_year()
    fig_by_category()
    fig_calibration()
    fig_forecast()
    fig_limits()
    print("Done.")


if __name__ == "__main__":
    main()
