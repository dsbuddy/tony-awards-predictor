"""
Fit the model on all historical seasons and predict a target year.

For 2026 the season file has Tony NOMINEES (won=false for all, winner=null) plus
the already-announced 2026 precursor results. We build features and rank.

Output: a readable per-category prediction with the model's top pick, its
softmax probability (confidence), the runner-up, and which precursors backed
the pick - so every prediction is explainable, not a black box.

Usage:
  python3 predict.py 2026
"""
import json
import os
import sys

from validate import load_all_seasons, season_to_rows
from matcher import build_features
from model import NaiveBayesTonyModel
from model_ensemble import EnsembleTonyModel
from categories import TONY_CATEGORIES

RAW = os.path.join(os.path.dirname(__file__), "..", "data", "raw")


def predict_year(target_year):
    seasons = load_all_seasons()
    train_seasons = [s for s in seasons if s["year"] != target_year]
    target = next((s for s in seasons if s["year"] == target_year), None)
    if target is None:
        print(f"No season file for {target_year}. Create season_{target_year}.json "
              f"with nominees + precursors first.")
        return
    if not train_seasons:
        print("No training seasons available.")
        return

    train_rows = [r for s in train_seasons for r in season_to_rows(s)]
    # Ensemble (NB at half weight + logistic elastic-net + HistGB): best LOYO
    # accuracy (66%) and calibration (Brier 0.47 vs 0.55). See model_ensemble.py.
    model = EnsembleTonyModel(weights=(0.5, 1.0, 1.0)).fit(train_rows)

    target_rows = build_features(target, target["precursors"])
    by_cat = {}
    for r in target_rows:
        by_cat.setdefault(r.category, []).append(r)

    print("=" * 70)
    print(f"TONY {target_year} PREDICTIONS  (trained on {len(train_seasons)} seasons)")
    print("=" * 70)

    # calibrated probabilities shown alongside raw (see calibrate.py)
    from calibrate import get_calibrator
    cal = get_calibrator()

    results = {}
    for cat in TONY_CATEGORIES:
        rows = by_cat.get(cat, [])
        if not rows:
            continue
        pred = model.predict_category(rows)
        top, prob, _ = pred[0]
        cal_prob = cal.calibrate_category(pred)[0][1]
        runner = pred[1] if len(pred) > 1 else None
        backers = []
        if top.dd_won: backers.append("DramaDesk")
        if top.occ_won: backers.append("OCC")
        if top.dl_won: backers.append("DramaLeague")
        if top.nydcc_won: backers.append("NYDCC")
        if top.odds_favorite: backers.append("PunditFav")
        name = top.nominee_person or top.nominee_show
        show = f" - {top.nominee_show}" if top.nominee_person else ""
        conf = "LOCK " if prob > 0.7 else ("lean " if prob > 0.45 else "toss ")
        print(f"\n{TONY_CATEGORIES[cat]}")
        print(f"  [{conf}{prob*100:4.0f}%] {name}{show}   (calibrated {cal_prob*100:.0f}%)")
        print(f"         signals: {', '.join(backers) if backers else 'none'}")
        if runner:
            rt, rp, _ = runner
            rn = rt.nominee_person or rt.nominee_show
            print(f"         runner-up: {rn} ({rp*100:.0f}%)")
        results[cat] = {"pick": name, "show": top.nominee_show,
                        "prob": prob, "calibrated_prob": cal_prob,
                        "signals": backers}

    out_path = os.path.join(os.path.dirname(__file__), "..", "output",
                            f"predictions_{target_year}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {out_path}")
    print("\nLegend: LOCK >70% | lean 45-70% | toss <45% (within-category softmax)")
    return results


if __name__ == "__main__":
    predict_year(int(sys.argv[1]) if len(sys.argv) > 1 else 2026)
