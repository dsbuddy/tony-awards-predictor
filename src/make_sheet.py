"""
Render a clean, printable Markdown prediction sheet from the model output.

    python3 make_sheet.py 2026
"""
import json
import os
import sys

from categories import TONY_CATEGORIES

OUT = os.path.join(os.path.dirname(__file__), "..", "output")

# Display order: big races first, then acting, creative, design.
ORDER = [
    "best_play", "best_musical", "best_revival_play", "best_revival_musical",
    "best_book", "best_score",
    "lead_actor_play", "lead_actress_play", "lead_actor_musical", "lead_actress_musical",
    "feat_actor_play", "feat_actress_play", "feat_actor_musical", "feat_actress_musical",
    "direction_play", "direction_musical", "choreography", "orchestrations",
    "scenic_play", "costume_play", "lighting_play", "sound_play",
    "scenic_musical", "costume_musical", "lighting_musical", "sound_musical",
]


def tier(p):
    if p > 0.70:
        return "LOCK"
    if p > 0.45:
        return "LEAN"
    return "TOSS-UP"


def make(year):
    with open(os.path.join(OUT, f"predictions_{year}.json")) as f:
        preds = json.load(f)

    lines = []
    lines.append(f"# Tony Award Predictions - {year}")
    lines.append("")
    lines.append("Model-based forecast built from precursor-award results "
                 "(Drama Desk, Outer Critics Circle, Drama League, NY Drama "
                 "Critics' Circle), Tony nomination tallies, and each show's "
                 "pre-ceremony critical, commercial, and predicted-odds standing.")
    lines.append("")
    lines.append("**How to read the columns.** *Model %* is the raw within-category "
                 "score (it saturates near 100% when every signal agrees). "
                 "*Calibrated %* maps that onto how often such picks actually win "
                 "historically - it's the more honest probability. Tiers below use "
                 "the calibrated number:")
    lines.append("")
    lines.append("- **LOCK** (>70% calibrated): the model's most reliable picks.")
    lines.append("- **LEAN** (45-70%): a real edge, not a certainty.")
    lines.append("- **TOSS-UP** (<45%): genuinely too close to call - the signals diverge.")
    lines.append("")
    lines.append("| Category | Predicted Winner | Production | Tier | Model % | Calibrated % |")
    lines.append("|---|---|---|---|---|---|")

    def cal_of(p):
        return p.get("calibrated_prob", p["prob"])

    for cat in ORDER:
        if cat not in preds:
            continue
        p = preds[cat]
        label = TONY_CATEGORIES[cat]
        pick = p["pick"]
        show = p["show"] if p.get("show") and p["show"] != pick else ""
        lines.append(f"| {label} | **{pick}** | {show} | {tier(cal_of(p))} | "
                     f"{p['prob']*100:.0f}% | {cal_of(p)*100:.0f}% |")

    # summary counts (by calibrated tier)
    locks = sum(1 for c in ORDER if c in preds and cal_of(preds[c]) > 0.70)
    leans = sum(1 for c in ORDER if c in preds and 0.45 < cal_of(preds[c]) <= 0.70)
    toss = sum(1 for c in ORDER if c in preds and cal_of(preds[c]) <= 0.45)
    lines.append("")
    lines.append(f"**Summary:** {locks} locks | {leans} leans | {toss} toss-ups "
                 f"across {locks+leans+toss} categories.")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("### Caveats worth knowing")
    lines.append("")
    lines.append("- **Best Musical** scores as a lock, but the pre-ceremony "
                 "consensus had it as a genuine three-way race. Treat it as the "
                 "softest of the locks.")
    lines.append("- The model runs a little **over-confident in the middle of "
                 "the range** - read a 95% as \"very likely,\" not literal odds.")
    lines.append("- Commercial/critical/odds inputs are tracked at the "
                 "**show level**, so a show sweeping the nominations lifts all "
                 "of its nominees.")
    lines.append("")

    path = os.path.join(OUT, f"PREDICTIONS_{year}.md")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote {path}")


if __name__ == "__main__":
    make(int(sys.argv[1]) if len(sys.argv) > 1 else 2026)
