"""
Category-specific deep-dive charts + the build-up (animated) frame sequences +
thumbnail concepts. Part 3 of the video asset set.

    python3 video_deepdive.py

Adds (continuing the numbering):
  16  acting vs production: which category TYPES are predictable
  17  the famous upsets the model would also have missed (humility + drama)
  18  Best Musical, year by year: hit/miss timeline
  build-up sequences: sweep (09a-c), climb (12a-d), forecast (06a-b) - numbered
       frames that reveal step by step for a screen-recorded animation
  thumb_A / thumb_B  thumbnail concepts
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import analytics
import video_analyses as VA

OUT = os.path.join(os.path.dirname(__file__), "..", "output", "video")
ANIM = os.path.join(OUT, "buildup")
os.makedirs(ANIM, exist_ok=True)

BG = "#0f1117"; PANEL = "#171a21"; INK = "#f5f6fa"; MUTE = "#9aa3b2"
GOLD = "#e8b339"; GOOD = "#3fb950"; MID = "#e3a008"; BAD = "#f85149"
BLUE = "#4493f8"; PURP = "#a371f7"; GRID = "#2a2e37"
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": INK, "axes.labelcolor": INK, "xtick.color": MUTE, "ytick.color": MUTE,
    "axes.edgecolor": GRID, "grid.color": GRID, "font.size": 17, "font.family": "DejaVu Sans",
})
FIG = (12.8, 7.2)


def _save(fig, name, sub="."):
    d = ANIM if sub == "buildup" else OUT
    fig.savefig(os.path.join(d, name), dpi=200, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  wrote {'buildup/' if sub=='buildup' else ''}{name}")


def _title(ax, text, sub=None):
    ax.text(0, 1.11 if sub else 1.05, text, transform=ax.transAxes, color=INK,
            fontsize=23, fontweight="bold", ha="left", va="bottom")
    if sub:
        ax.text(0, 1.045, sub, transform=ax.transAxes, color=MUTE, fontsize=15, ha="left", va="bottom")


# ---- 16: category-type predictability ----
def fig_group():
    g = VA.group_accuracy()
    order = ["production", "creative", "acting", "design"]
    names = {"production": "Production\n(Play, Musical, Revivals)", "creative": "Creative\n(Directing, Choreo)",
             "acting": "Acting\n(8 categories)", "design": "Design + Sound\n(8 categories)"}
    labels = [names[k] for k in order]
    vals = [g[k] * 100 for k in order]
    colors = [GOOD if v >= 65 else (MID if v >= 58 else BAD) for v in vals]
    fig, ax = plt.subplots(figsize=FIG)
    fig.subplots_adjust(top=0.82, bottom=0.2)
    x = np.arange(len(labels))
    ax.bar(x, vals, color=colors, width=0.62, zorder=3)
    for i, v in enumerate(vals):
        ax.text(i, v + 2, f"{v:.0f}%", ha="center", color=INK, fontsize=24, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=14)
    ax.set_ylim(0, 90); ax.set_ylabel("Accuracy (%)")
    ax.grid(axis="y", alpha=0.4, zorder=0)
    _title(ax, "Not all categories are created equal",
           "The big production races are nearly solved. Design is a near coin-flip.")
    _save(fig, "16_group_types.png")


# ---- 17: the famous upsets even the model misses ----
def fig_upsets():
    # hand-curated from the live miss analysis (real, recognizable upsets)
    upsets = [
        ("2004 Best Musical", "Wicked", "Avenue Q", 84),
        ("2018 Best Musical Revival", "My Fair Lady", "Once on This Island", 82),
        ("2009 Best Score", "Billy Elliot", "Next to Normal", 91),
        ("2021 Lead Actress Play", "Joaquina Kalukango", "Mary-Louise Parker", 81),
    ]
    fig, ax = plt.subplots(figsize=FIG)
    fig.subplots_adjust(top=0.8, left=0.30)
    y = np.arange(len(upsets))
    confs = [u[3] for u in upsets]
    ax.barh(y, confs, color=BAD, alpha=0.85, zorder=3)
    for i, (cat, pick, won, c) in enumerate(upsets):
        ax.text(c - 2, i, f"model said {pick}", va="center", ha="right", color=BG, fontsize=12.5, fontweight="bold")
        ax.text(c + 1.5, i, f"-> {won} won", va="center", color=GOOD, fontsize=13, fontweight="bold")
    ax.set_yticks(y); ax.set_yticklabels([u[0] for u in upsets], fontsize=13)
    ax.invert_yaxis(); ax.set_xlim(0, 116); ax.set_xlabel("Model's (wrong) confidence (%)")
    ax.grid(axis="x", alpha=0.4, zorder=0)
    _title(ax, "The upsets even the model walked into",
           "Avenue Q over Wicked. The data was sure. The voters had other plans.")
    _save(fig, "17_upsets.png")


# ---- 18: Best Musical hit/miss timeline ----
def fig_bestmusical():
    d = VA.category_track_record("best_musical")
    fig, ax = plt.subplots(figsize=FIG)
    fig.subplots_adjust(top=0.82, bottom=0.2)
    years = [str(y) for y, _ in d]
    x = np.arange(len(years))
    for i, (_, ok) in enumerate(d):
        ax.scatter([i], [0], s=480, color=GOOD if ok else BAD, zorder=3, edgecolor=BG, linewidth=2)
        ax.text(i, 0, "OK" if ok else "X", ha="center", va="center", color=BG, fontsize=11, fontweight="bold")
    hits = sum(1 for _, ok in d if ok)
    ax.set_xticks(x); ax.set_xticklabels(years, rotation=45, fontsize=11)
    ax.set_yticks([]); ax.set_ylim(-1, 1)
    ax.spines[["left", "right", "top"]].set_visible(False)
    _title(ax, f"Best Musical: {hits} of {len(d)} right",
           "Tougher than Best Play - the splashy category throws more curveballs")
    _save(fig, "18_bestmusical_track.png")


# ---- BUILD-UP: sweep effect revealed bar by bar (09a, 09b, 09c) ----
def buildup_sweep():
    d = VA.sweep_effect()
    order = ["won both DD + OCC", "won just one", "won neither"]
    labels = ["Won BOTH\nDD + Outer Critics", "Won just\none", "Won\nneither"]
    vals = [d[k][2] * 100 for k in order]
    colors = [GOOD, MID, BAD]
    for step in range(1, 4):
        fig, ax = plt.subplots(figsize=FIG)
        fig.subplots_adjust(top=0.82, bottom=0.18)
        x = np.arange(3)
        shown = [vals[i] if i < step else 0 for i in range(3)]
        ax.bar(x, shown, color=[colors[i] if i < step else PANEL for i in range(3)], width=0.62, zorder=3)
        for i in range(step):
            ax.text(i, vals[i] + 3, f"{vals[i]:.0f}%", ha="center", color=INK, fontsize=26, fontweight="bold")
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=15)
        ax.set_ylim(0, 100); ax.set_ylabel("Chance of winning the Tony (%)")
        ax.grid(axis="y", alpha=0.4, zorder=0)
        _title(ax, "Sweep the big two, and it's basically over",
               "Revealed step by step for animation")
        _save(fig, f"09{'abc'[step-1]}_sweep_step{step}.png", sub="buildup")


# ---- BUILD-UP: the climb, point by point (12a-d) ----
def buildup_climb():
    d = VA.ablation_climb()
    labels = [n for n, _ in d]
    vals = [a * 100 for _, a in d]
    x = np.arange(len(labels))
    for step in range(1, len(labels) + 1):
        fig, ax = plt.subplots(figsize=FIG)
        fig.subplots_adjust(top=0.82, bottom=0.2)
        ax.plot(x[:step], vals[:step], color=GOLD, lw=3, marker="o", markersize=14, zorder=3)
        if step > 1:
            ax.fill_between(x[:step], 50, vals[:step], color=GOLD, alpha=0.12, zorder=1)
        for i in range(step):
            ax.text(x[i], vals[i] + 1.3, f"{vals[i]:.0f}%", ha="center", color=INK, fontsize=18, fontweight="bold")
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=14)
        ax.set_ylim(50, 70); ax.set_ylabel("Accuracy (%)")
        ax.grid(axis="y", alpha=0.4, zorder=0)
        _title(ax, "How each layer of data moves the needle",
               "Revealed step by step for animation")
        _save(fig, f"12{'abcd'[step-1]}_climb_step{step}.png", sub="buildup")


# ---- THUMBNAILS ----
def thumbnail_A():
    o = analytics.overall()
    fig, ax = plt.subplots(figsize=(12.8, 7.2))
    ax.axis("off")
    ax.text(0.5, 0.72, "I PREDICTED", ha="center", fontsize=46, fontweight="bold", color=INK)
    ax.text(0.5, 0.5, "THE TONYS", ha="center", fontsize=72, fontweight="bold", color=GOLD)
    ax.text(0.5, 0.30, "before they happened", ha="center", fontsize=30, color=INK, style="italic")
    ax.text(0.5, 0.13, f"{o['acc']*100:.0f}% with math", ha="center", fontsize=26, color=GOOD, fontweight="bold")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    _save(fig, "thumb_A.png")


def thumbnail_B():
    fig, ax = plt.subplots(figsize=(12.8, 7.2))
    ax.axis("off")
    ax.text(0.5, 0.78, "CAN MATH PREDICT", ha="center", fontsize=40, fontweight="bold", color=INK)
    ax.text(0.5, 0.60, "THE TONY AWARDS?", ha="center", fontsize=52, fontweight="bold", color=GOLD)
    # mini sweep bars as the visual
    for i, (v, c) in enumerate([(76, GOOD), (38, MID), (9, BAD)]):
        ax.add_patch(plt.Rectangle((0.30 + i * 0.16, 0.12), 0.11, v / 100 * 0.34, color=c))
        ax.text(0.355 + i * 0.16, 0.12 + v / 100 * 0.34 + 0.02, f"{v}%", ha="center", color=INK, fontsize=20, fontweight="bold")
    ax.text(0.5, 0.07, "the answer surprised me", ha="center", fontsize=20, color=MUTE, style="italic")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    _save(fig, "thumb_B.png")


def main():
    print(f"Rendering deep-dive + build-up + thumbnails -> {os.path.relpath(OUT)}/")
    fig_group()
    fig_upsets()
    fig_bestmusical()
    buildup_sweep()
    buildup_climb()
    thumbnail_A()
    thumbnail_B()
    print("Done.")


if __name__ == "__main__":
    main()
