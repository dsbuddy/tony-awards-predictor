"""
Expanded video chart library (part 2).

A data-storytelling video wants a fresh visual every 10-15 seconds, so this adds
a deeper set on top of video_figures.py's 7 core frames. Same dark house style.
Every chart is backed by a real analysis in video_analyses.py.

    python3 video_figures2.py

Output: output/video/*.png (continues the numbering at 08+).
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import analytics
import video_analyses as VA

OUT = os.path.join(os.path.dirname(__file__), "..", "output", "video")
os.makedirs(OUT, exist_ok=True)

BG = "#0f1117"; PANEL = "#171a21"; INK = "#f5f6fa"; MUTE = "#9aa3b2"
GOLD = "#e8b339"; GOOD = "#3fb950"; MID = "#e3a008"; BAD = "#f85149"
BLUE = "#4493f8"; PURP = "#a371f7"; TEAL = "#39c5cf"; GRID = "#2a2e37"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": INK, "axes.labelcolor": INK, "xtick.color": MUTE, "ytick.color": MUTE,
    "axes.edgecolor": GRID, "grid.color": GRID, "font.size": 17, "font.family": "DejaVu Sans",
})
FIG = (12.8, 7.2)


def _save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=200, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  wrote {name}")


def _title(ax, text, sub=None):
    ax.text(0, 1.11 if sub else 1.05, text, transform=ax.transAxes, color=INK,
            fontsize=23, fontweight="bold", ha="left", va="bottom")
    if sub:
        ax.text(0, 1.045, sub, transform=ax.transAxes, color=MUTE, fontsize=15,
                ha="left", va="bottom")


def _bars_labeled(ax, labels, vals, colors, fmt="{:.0f}%", fs=15):
    y = np.arange(len(labels))
    ax.barh(y, vals, color=colors, zorder=3)
    ax.set_yticks(y); ax.set_yticklabels(labels)
    ax.invert_yaxis()
    for i, v in enumerate(vals):
        ax.text(v + max(vals) * 0.015, i, fmt.format(v), va="center", color=INK, fontsize=fs, fontweight="bold")


# 08: precursor power ranking (the kingmakers)
def fig_precursor_power():
    d = VA.precursor_power()
    items = sorted(d.items(), key=lambda kv: kv[1][2], reverse=True)
    labels = [k for k, _ in items]
    vals = [v[2] * 100 for _, v in items]
    colors = [GOLD if l == "Pundit favorite" else BLUE for l in labels]
    fig, ax = plt.subplots(figsize=FIG)
    fig.subplots_adjust(top=0.84, left=0.26)
    _bars_labeled(ax, labels, vals, colors)
    ax.set_xlim(0, 100); ax.set_xlabel("How often its pick wins the Tony (%)")
    ax.grid(axis="x", alpha=0.4, zorder=0)
    _title(ax, "Which early award is the kingmaker?",
           "Follow just ONE award's pick - here's how often you'd be right")
    _save(fig, "08_precursor_power.png")


# 09: the sweep effect (the single best chart)
def fig_sweep():
    d = VA.sweep_effect()
    order = ["won both DD + OCC", "won just one", "won neither"]
    labels = ["Won BOTH\nDrama Desk + Outer Critics", "Won just\none of them", "Won\nneither"]
    vals = [d[k][2] * 100 for k in order]
    colors = [GOOD, MID, BAD]
    fig, ax = plt.subplots(figsize=FIG)
    fig.subplots_adjust(top=0.82, bottom=0.18)
    x = np.arange(len(labels))
    ax.bar(x, vals, color=colors, width=0.62, zorder=3)
    for i, (k, v) in enumerate(zip(order, vals)):
        ax.text(i, v + 3.5, f"{v:.0f}%", ha="center", color=INK, fontsize=26, fontweight="bold")
        # n-count sits just inside the top of each bar (no collision with x labels)
        ax.text(i, max(v - 6, 4), f"n={d[k][1]}", ha="center", color=BG, fontsize=13, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=15)
    ax.set_ylim(0, 100); ax.set_ylabel("Chance of winning the Tony (%)")
    ax.grid(axis="y", alpha=0.4, zorder=0)
    _title(ax, "Sweep the big two, and it's basically over",
           "A nominee's Tony odds, by how many of the two major early awards they won")
    _save(fig, "09_sweep_effect.png")


# 10: agreement -> accuracy
def fig_agreement():
    d = VA.agreement_accuracy()
    ks = [k for k in d if d[k][1] >= 3]
    labels = [f"{k}" for k in ks]
    vals = [d[k][2] * 100 for k in ks]
    fig, ax = plt.subplots(figsize=FIG)
    fig.subplots_adjust(top=0.82, bottom=0.16)
    x = np.arange(len(labels))
    ax.plot(x, vals, color=GOLD, lw=3, marker="o", markersize=14, zorder=3)
    for i, k in enumerate(ks):
        ax.text(x[i], vals[i] + 5, f"{vals[i]:.0f}%", ha="center", color=INK, fontsize=18, fontweight="bold")
        ax.text(x[i], vals[i] - 7, f"n={d[k][1]}", ha="center", color=MUTE, fontsize=12)
    ax.set_xticks(x); ax.set_xticklabels([f"{k} award(s)" for k in ks], fontsize=14)
    ax.set_ylim(0, 112); ax.set_ylabel("Consensus pick wins the Tony (%)")
    ax.set_xlabel("Number of early awards backing the same nominee")
    ax.grid(alpha=0.4, zorder=0)
    _title(ax, "The more they agree, the surer the bet",
           "When all the early awards line up behind one name, it almost never loses")
    _save(fig, "10_agreement.png")


# 11: model vs pundits (the humbling honesty beat)
def fig_model_vs_pundits():
    d = VA.model_vs_pundits()
    fig, ax = plt.subplots(figsize=FIG)
    fig.subplots_adjust(top=0.82, bottom=0.2)
    labels = ["The model\n(all 612 races)", "Pro pundits\n(220 marquee races)"]
    vals = [d["model"][2] * 100, d["pundits"][2] * 100]
    ax.bar([0, 1], vals, color=[GOLD, PURP], width=0.55, zorder=3)
    for i, v in enumerate(vals):
        ax.text(i, v + 2, f"{v:.0f}%", ha="center", color=INK, fontsize=28, fontweight="bold")
    ax.set_xticks([0, 1]); ax.set_xticklabels(labels, fontsize=16)
    ax.set_ylim(0, 100); ax.set_ylabel("Accuracy (%)")
    ax.grid(axis="y", alpha=0.4, zorder=0)
    _title(ax, "Can a model beat the human experts?",
           "Honest answer: on the big races they bother to call, the pundits still win")
    _save(fig, "11_model_vs_pundits.png")


# 12: the climb (ablation - each signal adds accuracy)
def fig_climb():
    d = VA.ablation_climb()
    labels = [n for n, _ in d]
    vals = [a * 100 for _, a in d]
    fig, ax = plt.subplots(figsize=FIG)
    fig.subplots_adjust(top=0.82, bottom=0.2)
    x = np.arange(len(labels))
    ax.plot(x, vals, color=GOLD, lw=3, marker="o", markersize=14, zorder=3)
    ax.fill_between(x, 50, vals, color=GOLD, alpha=0.12, zorder=1)
    for i, v in enumerate(vals):
        ax.text(x[i], v + 1.3, f"{v:.0f}%", ha="center", color=INK, fontsize=18, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=14)
    ax.set_ylim(50, 70); ax.set_ylabel("Accuracy (%)")
    ax.grid(axis="y", alpha=0.4, zorder=0)
    _title(ax, "How each layer of data moves the needle",
           "Start with the early awards, then stack on everything else knowable beforehand")
    _save(fig, "12_climb.png")


# 13: Best Play track record (the "solved" category, year by year)
def fig_bestplay_track():
    d = VA.category_track_record("best_play")
    fig, ax = plt.subplots(figsize=FIG)
    fig.subplots_adjust(top=0.82, bottom=0.2)
    years = [str(y) for y, _ in d]
    x = np.arange(len(years))
    for i, (_, ok) in enumerate(d):
        ax.scatter([i], [0], s=520, color=GOOD if ok else BAD, zorder=3, edgecolor=BG, linewidth=2)
        ax.text(i, 0.0, "OK" if ok else "X", ha="center", va="center", color=BG, fontsize=12, fontweight="bold")
    hits = sum(1 for _, ok in d if ok)
    ax.set_xticks(x); ax.set_xticklabels(years, rotation=45, fontsize=11)
    ax.set_yticks([]); ax.set_ylim(-1, 1)
    ax.spines[["left", "right", "top"]].set_visible(False)
    _title(ax, f"Best Play: called {hits} of {len(d)} right",
           "Green = the model nailed it. The single most predictable category.")
    _save(fig, "13_bestplay_track.png")


# 14: confidence vs outcome scatter (every single race)
def fig_scatter():
    rs = analytics.confidence_vs_correct()
    fig, ax = plt.subplots(figsize=FIG)
    fig.subplots_adjust(top=0.84)
    hit = [(r["prob"] * 100, r["year"]) for r in rs if r["correct"]]
    miss = [(r["prob"] * 100, r["year"]) for r in rs if not r["correct"]]
    if hit:
        ax.scatter(*zip(*hit), c=GOOD, alpha=0.55, s=42, label="Called it", zorder=3)
    if miss:
        ax.scatter(*zip(*miss), c=BAD, alpha=0.55, s=42, marker="x", label="Missed", zorder=3)
    ax.axvline(70, color=MUTE, ls=":", lw=1.5)
    ax.text(71, min(r["year"] for r in rs), " lock line", color=MUTE, fontsize=12)
    ax.set_xlabel("Model confidence (%)"); ax.set_ylabel("Season")
    ax.grid(alpha=0.35, zorder=0)
    ax.legend(loc="lower left", facecolor=PANEL, edgecolor=GRID, fontsize=13)
    _title(ax, "Every prediction, 26 seasons at once",
           "Misses cluster at low confidence - exactly where they should")
    _save(fig, "14_scatter.png")


# 15: the funnel - 5 nominees down to 1, model's edge
def fig_funnel():
    o = analytics.overall()
    fig, ax = plt.subplots(figsize=FIG)
    ax.axis("off")
    tiers = [("Random guess (5 nominees)", 20, BAD),
             ("Follow the most early-award wins", round(o["baseline_acc"] * 100), MID),
             ("The full model", round(o["acc"] * 100), GOLD)]
    for i, (label, v, c) in enumerate(tiers):
        w = v / 100 * 0.8
        y = 0.7 - i * 0.22
        ax.add_patch(plt.Rectangle((0.5 - w / 2, y), w, 0.14, color=c, zorder=3))
        ax.text(0.5, y + 0.07, f"{v}%", ha="center", va="center", color=BG, fontsize=22, fontweight="bold")
        ax.text(0.5, y - 0.04, label, ha="center", va="center", color=INK, fontsize=15)
    ax.text(0.5, 0.93, "From a coin flip to two-in-three", ha="center", fontsize=24, fontweight="bold", color=INK)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    _save(fig, "15_funnel.png")


def main():
    print(f"Rendering expanded video figures -> {os.path.relpath(OUT)}/")
    fig_precursor_power()
    fig_sweep()
    fig_agreement()
    fig_model_vs_pundits()
    fig_climb()
    fig_bestplay_track()
    fig_scatter()
    fig_funnel()
    print("Done.")


if __name__ == "__main__":
    main()
