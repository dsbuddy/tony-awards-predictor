"""
Leave-one-year-out backtesting harness.

This is the honesty check. For each historical season Y:
  * train the model on every season EXCEPT Y
  * predict each category in Y
  * record whether the model's top pick actually won

We report:
  * overall top-1 hit rate (the headline number)
  * per-category hit rate (which categories are predictable vs coin-flips)
  * per-category-group hit rate
  * calibration: when the model says "70% likely", does it win ~70% of the time?
  * a baseline for comparison: "pick whoever won the most precursors" (naive)

Leave-one-YEAR-out (not leave-one-row-out) is the correct CV here because rows
within a year are not independent (they compete), and because in production we
always predict a whole unseen year at once.
"""

import json
import os
from collections import defaultdict

from schema import FeatureRow
from matcher import build_features
from model import NaiveBayesTonyModel, CATEGORY_GROUP
from model_ensemble import EnsembleTonyModel
from categories import TONY_CATEGORIES

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")


def load_all_seasons():
    seasons = []
    for fn in sorted(os.listdir(RAW_DIR)):
        if fn.startswith("season_") and fn.endswith(".json"):
            with open(os.path.join(RAW_DIR, fn)) as f:
                seasons.append(json.load(f))
    return seasons


def season_to_rows(season):
    return build_features(season, season["precursors"])


def _dictrow_to_feature(d):
    fr = FeatureRow(year=d["year"], category=d["category"], nominee_key=d["nominee_key"],
                    nominee_show=d["nominee_show"], nominee_person=d.get("nominee_person"))
    for k, v in d.items():
        if hasattr(fr, k):
            setattr(fr, k, v)
    return fr


def baseline_pick(rows_in_cat):
    """Naive baseline: pick the nominee with the most precursor WINS, breaking
    ties by most precursor nominations. This is what a casual fan does."""
    # Sort by a leak-free key first so ties don't resolve to source order
    # (Wikipedia lists winners first). Same fix as the model.
    best, best_key = None, (-1.0, -1.0, "")
    for r in sorted(rows_in_cat, key=lambda r: r.nominee_key):
        # negative nominee_key rank can't be compared; use precursor counts only,
        # and on a tie keep the alphabetically-first (deterministic, leak-free)
        k = (r.n_precursor_wins, r.n_precursor_noms)
        if (k[0], k[1]) > (best_key[0], best_key[1]):
            best_key, best = (k[0], k[1], r.nominee_key), r
    return best


def backtest(pool_by_group=True, verbose=True):
    seasons = load_all_seasons()
    if len(seasons) < 2:
        print(f"Need >=2 seasons to backtest; found {len(seasons)}. "
              f"Gather pilot data first.")
        return None

    all_rows_by_year = {s["year"]: season_to_rows(s) for s in seasons}
    years = sorted(all_rows_by_year)

    model_hits = defaultdict(lambda: [0, 0])     # category -> [hits, total]
    base_hits = defaultdict(lambda: [0, 0])
    group_hits = defaultdict(lambda: [0, 0])
    calib = []  # (predicted_prob_of_top_pick, did_top_pick_win)
    overall = [0, 0]
    base_overall = [0, 0]

    for held in years:
        train = [r for y in years if y != held for r in all_rows_by_year[y]]
        model = EnsembleTonyModel(weights=(0.5, 1.0, 1.0)).fit(train)

        test_rows = all_rows_by_year[held]
        by_cat = defaultdict(list)
        for r in test_rows:
            by_cat[r.category].append(r)

        for cat, rows in by_cat.items():
            if not rows or all(r.won_tony is None for r in rows):
                continue  # no labeled winner this year/category -> skip
            if not any(r.won_tony and r.won_tony >= 0.5 for r in rows):
                continue
            # model prediction
            pred = model.predict_category(rows)
            top, prob, _ = pred[0]
            hit = 1 if (top.won_tony and top.won_tony >= 0.5) else 0
            model_hits[cat][0] += hit; model_hits[cat][1] += 1
            group_hits[CATEGORY_GROUP.get(cat, "?")][0] += hit
            group_hits[CATEGORY_GROUP.get(cat, "?")][1] += 1
            overall[0] += hit; overall[1] += 1
            calib.append((prob, hit))
            # baseline prediction
            b = baseline_pick(rows)
            bhit = 1 if (b.won_tony and b.won_tony >= 0.5) else 0
            base_hits[cat][0] += bhit; base_hits[cat][1] += 1
            base_overall[0] += bhit; base_overall[1] += 1

    if verbose:
        _print_report(overall, base_overall, group_hits, model_hits, base_hits, calib, years)
    return {
        "overall": overall, "baseline": base_overall,
        "by_category": dict(model_hits), "by_group": dict(group_hits),
        "calibration": calib, "years": years,
    }


def _rate(ht):
    h, t = ht
    return f"{h}/{t} = {100*h/t:4.0f}%" if t else "  n/a"


def _print_report(overall, base_overall, group_hits, model_hits, base_hits, calib, years):
    print("=" * 64)
    print(f"BACKTEST over {len(years)} seasons: {years[0]}-{years[-1]}")
    print("=" * 64)
    print(f"\nOVERALL  model: {_rate(overall)}    baseline(most-precursors): {_rate(base_overall)}")
    lift = (overall[0]/overall[1] - base_overall[0]/base_overall[1]) * 100 if overall[1] and base_overall[1] else 0
    print(f"         model lift over baseline: {lift:+.1f} pts")
    print("\nBY GROUP:")
    for g in sorted(group_hits):
        print(f"  {g:12s} {_rate(group_hits[g])}")
    print("\nBY CATEGORY (model vs baseline):")
    for cat in TONY_CATEGORIES:
        if cat in model_hits:
            print(f"  {cat:22s} model {_rate(model_hits[cat]):14s}  base {_rate(base_hits[cat])}")
    # calibration in 4 buckets
    print("\nCALIBRATION (predicted prob of top pick vs actual win rate):")
    buckets = defaultdict(lambda: [0.0, 0])
    for p, hit in calib:
        b = min(int(p * 4), 3)
        buckets[b][0] += hit; buckets[b][1] += 1
    labels = ["0-25%", "25-50%", "50-75%", "75-100%"]
    for b in range(4):
        h, n = buckets[b]
        if n:
            print(f"  {labels[b]:8s} predicted -> actual {100*h/n:4.0f}%  (n={n})")


if __name__ == "__main__":
    backtest()
