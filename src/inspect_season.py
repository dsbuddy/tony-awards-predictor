"""
Pipeline diagnostic: load a season, build features, and show - per category - which Tony winner got which precursor signals. This is how we confirm the
crosswalk + matcher actually fire before trusting any backtest numbers.

Run: python3 inspect_season.py 2025
"""
import json, os, sys
from matcher import build_features
from categories import TONY_CATEGORIES

RAW = os.path.join(os.path.dirname(__file__), "..", "data", "raw")


def main(year):
    with open(os.path.join(RAW, f"season_{year}.json")) as f:
        season = json.load(f)
    rows = build_features(season, season["precursors"])
    by_cat = {}
    for r in rows:
        by_cat.setdefault(r.category, []).append(r)

    matched_winner_signals = 0
    winners_total = 0
    print(f"{'CATEGORY':24s} {'TONY WINNER':28s} DD  OCC DL  NY")
    print("-" * 78)
    for cat in TONY_CATEGORIES:
        rs = by_cat.get(cat, [])
        win = next((r for r in rs if r.won_tony and r.won_tony >= 0.5), None)
        if not win:
            print(f"{cat:24s} (no winner labeled)")
            continue
        winners_total += 1
        dd = "W" if win.dd_won else ("n" if win.dd_nominated else "-")
        occ = "W" if win.occ_won else ("n" if win.occ_nominated else "-")
        dl = "W" if win.dl_won else ("n" if win.dl_nominated else "-")
        ny = "W" if win.nydcc_won else "-"
        if win.n_precursor_wins > 0:
            matched_winner_signals += 1
        name = (win.nominee_person or win.nominee_show)[:27]
        print(f"{cat:24s} {name:28s} {dd:3s} {occ:3s} {dl:3s} {ny}")
    print("-" * 78)
    print(f"Tony winners that won >=1 precursor: {matched_winner_signals}/{winners_total}")
    print("(High overlap = matcher is firing. OCC all '-' = OCC data not yet entered.)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "2025")
