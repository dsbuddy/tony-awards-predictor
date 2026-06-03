"""
Generate the chart set for the project, written to output/charts/.

    python3 make_charts.py

Produces:
  accuracy_by_year.png      model vs baseline, season by season
  accuracy_by_category.png  which categories are predictable vs coin-flips
  accuracy_by_group.png     production / acting / creative / design
  calibration.png           predicted confidence vs actual win rate
  confidence_scatter.png    every race: confidence vs hit/miss
  forecast_2026.png         the 2026 predictions ranked by confidence

Pure matplotlib (no seaborn dependency required at runtime).
"""
import os

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

import analytics

OUT = os.path.join(os.path.dirname(__file__), "..", "output", "charts")
os.makedirs(OUT, exist_ok=True)

# a calm, consistent palette
C_MODEL = "#2b6cb0"
C_BASE = "#cbd5e0"
C_GOOD = "#2f855a"
C_MID = "#d69e2e"
C_BAD = "#c53030"
GROUP_COLORS = {"production": "#2b6cb0", "acting": "#805ad5",
                "creative": "#319795", "design": "#dd6b20", "?": "#718096"}


def _save(fig, name):
    path = os.path.join(OUT, name)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {os.path.relpath(path)}")


def chart_accuracy_by_year():
    data = analytics.per_year_accuracy()
    years = [str(d["year"]) for d in data]
    acc = [d["acc"] * 100 for d in data]
    base = [d["baseline_acc"] * 100 for d in data]
    overall = analytics.overall()["acc"] * 100

    fig, ax = plt.subplots(figsize=(11, 5))
    x = range(len(years))
    ax.bar([i - 0.2 for i in x], acc, width=0.4, label="Model", color=C_MODEL)
    ax.bar([i + 0.2 for i in x], base, width=0.4, label="Baseline (most precursors)",
           color=C_BASE)
    ax.axhline(overall, color=C_BAD, ls="--", lw=1.2,
               label=f"Model average ({overall:.0f}%)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(years, rotation=45)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Prediction accuracy by season (leave-one-year-out)")
    ax.set_ylim(0, 100)
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)
    _save(fig, "accuracy_by_year.png")


def chart_accuracy_by_category():
    data = analytics.per_category_accuracy()
    labels = [d["label"] for d in data]
    acc = [d["acc"] * 100 for d in data]
    colors = [GROUP_COLORS.get(d["group"], "#718096") for d in data]

    fig, ax = plt.subplots(figsize=(9, 9))
    y = range(len(labels))
    ax.barh(list(y), acc, color=colors)
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Accuracy (%)")
    ax.set_xlim(0, 100)
    ax.set_title("Which categories are predictable? (by precursor signal strength)")
    ax.grid(axis="x", alpha=0.3)
    # legend for groups
    from matplotlib.patches import Patch
    handles = [Patch(color=c, label=g.title())
               for g, c in GROUP_COLORS.items() if g != "?"]
    ax.legend(handles=handles, loc="lower right", framealpha=0.9)
    _save(fig, "accuracy_by_category.png")


def chart_accuracy_by_group():
    data = analytics.per_group_accuracy()
    groups = [d["group"].title() for d in data]
    acc = [d["acc"] * 100 for d in data]
    colors = [GROUP_COLORS.get(d["group"], "#718096") for d in data]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(groups, acc, color=colors)
    for i, d in enumerate(data):
        ax.text(i, d["acc"] * 100 + 1.5, f"{d['acc']*100:.0f}%\n({d['hits']}/{d['total']})",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Accuracy by category type")
    ax.grid(axis="y", alpha=0.3)
    _save(fig, "accuracy_by_group.png")


def chart_calibration():
    data = analytics.calibration_bins(nbins=10)
    mid = [d["predicted_mid"] * 100 for d in data]
    actual = [d["actual"] * 100 for d in data]
    sizes = [max(30, d["n"] * 6) for d in data]

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.plot([0, 100], [0, 100], ls="--", color="#a0aec0", label="Perfect calibration")
    ax.scatter(mid, actual, s=sizes, color=C_MODEL, alpha=0.75, zorder=3,
               label="Confidence bin (size = # races)")
    ax.set_xlabel("Model confidence (%)")
    ax.set_ylabel("Actual win rate (%)")
    ax.set_title("Calibration: does the stated confidence hold up?")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(alpha=0.3)
    _save(fig, "calibration.png")


def chart_confidence_scatter():
    data = analytics.confidence_vs_correct()
    hit = [d["prob"] * 100 for d in data if d["correct"]]
    hit_y = [d["year"] for d in data if d["correct"]]
    miss = [d["prob"] * 100 for d in data if not d["correct"]]
    miss_y = [d["year"] for d in data if not d["correct"]]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(hit, hit_y, color=C_GOOD, alpha=0.6, label="Correct", s=35)
    ax.scatter(miss, miss_y, color=C_BAD, alpha=0.6, label="Missed", s=35, marker="x")
    ax.axvline(70, color="#718096", ls=":", lw=1, label="Lock threshold (70%)")
    ax.set_xlabel("Model confidence (%)")
    ax.set_ylabel("Season")
    ax.set_title("Every prediction: confidence vs. outcome")
    ax.legend(loc="lower left", framealpha=0.9)
    ax.grid(alpha=0.3)
    _save(fig, "confidence_scatter.png")


def chart_forecast(year=2026):
    data = analytics.prediction_confidences(year)
    if not data:
        print(f"  (no season data for {year}, skipping forecast chart)")
        return
    labels = [f"{d['label']}" for d in data]
    probs = [d["prob"] * 100 for d in data]

    def tier_color(p):
        return C_GOOD if p > 70 else (C_MID if p > 45 else C_BAD)
    colors = [tier_color(p) for p in probs]

    fig, ax = plt.subplots(figsize=(9, 10))
    y = range(len(labels))
    ax.barh(list(y), probs, color=colors)
    for i, d in enumerate(data):
        ax.text(min(d["prob"] * 100 + 1, 99), i, f" {d['pick']}",
                va="center", fontsize=8, color="#1a202c")
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Model confidence (%)")
    ax.set_xlim(0, 110)
    ax.set_title(f"{year} forecast, ranked by confidence")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=C_GOOD, label="Lock (>70%)"),
                       Patch(color=C_MID, label="Lean (45-70%)"),
                       Patch(color=C_BAD, label="Toss-up (<45%)")],
              loc="lower right", framealpha=0.9)
    ax.grid(axis="x", alpha=0.3)
    _save(fig, f"forecast_{year}.png")


def main():
    print(f"Generating charts -> {os.path.relpath(OUT)}/")
    chart_accuracy_by_year()
    chart_accuracy_by_category()
    chart_accuracy_by_group()
    chart_calibration()
    chart_confidence_scatter()
    chart_forecast(2026)
    print("Done.")


if __name__ == "__main__":
    main()
