"""
Compute structured accuracy/confidence metrics for the model, the single source
of truth that the charting and reporting scripts both build on.

Everything here uses the same leave-one-year-out (LOYO) protocol as validate.py:
for each season we train on every *other* season and predict the held-out one,
so no result ever leaks into its own prediction.

Public functions return plain dicts / lists of dicts (easy to feed to pandas):
  per_year_accuracy()      -> [{year, hits, total, acc, baseline_acc}]
  per_category_accuracy()  -> [{category, label, group, hits, total, acc, baseline_acc}]
  per_group_accuracy()     -> [{group, hits, total, acc}]
  calibration_bins()       -> [{bin_lo, bin_hi, predicted_mid, actual, n}]
  confidence_vs_correct()  -> [{year, category, prob, correct}]  (one row per race)
  prediction_confidences(year) -> [{category, label, group, pick, prob, tier}]
"""
from collections import defaultdict

from validate import load_all_seasons, season_to_rows, baseline_pick
from model import NaiveBayesTonyModel, CATEGORY_GROUP
from model_ensemble import EnsembleTonyModel
from categories import TONY_CATEGORIES


def _labeled_rows_by_year():
    """{year: [rows]} for seasons that actually have a recorded winner."""
    out = {}
    for s in load_all_seasons():
        rows = season_to_rows(s)
        if any(r.won_tony for r in rows):
            out[s["year"]] = rows
    return out


def _loyo_predictions():
    """Run the full LOYO backtest once and return a flat list of race outcomes:
    [{year, category, prob, correct, baseline_correct}]. Everything else is
    derived from this so we only pay for the backtest a single time."""
    rby = _labeled_rows_by_year()
    years = sorted(rby)
    races = []
    for held in years:
        train = [r for y in years if y != held for r in rby[y]]
        model = EnsembleTonyModel(weights=(0.5, 1.0, 1.0)).fit(train)
        bycat = defaultdict(list)
        for r in rby[held]:
            bycat[r.category].append(r)
        for cat, rows in bycat.items():
            if not any(x.won_tony for x in rows):
                continue
            pred = model.predict_category(rows)
            top, prob, _ = pred[0]
            base = baseline_pick(rows)
            races.append({
                "year": held,
                "category": cat,
                "prob": prob,
                "correct": bool(top.won_tony and top.won_tony >= 0.5),
                "baseline_correct": bool(base.won_tony and base.won_tony >= 0.5),
                # every nominee's (prob, won) for calibration fitting
                "all_probs": [(p, bool(r.won_tony and r.won_tony >= 0.5))
                              for r, p, _ in pred],
            })
    return races


def all_nominee_probs():
    """Flat list of (predicted_prob, won) over EVERY nominee across all LOYO
    races - used to fit the probability calibrator on full 0..1 support."""
    out = []
    for r in races():
        out.extend(r["all_probs"])
    return out


# cache so repeated calls in one process don't re-run the backtest
_RACES = None


def races():
    global _RACES
    if _RACES is None:
        _RACES = _loyo_predictions()
    return _RACES


def per_year_accuracy():
    agg = defaultdict(lambda: {"hits": 0, "total": 0, "base": 0})
    for r in races():
        a = agg[r["year"]]
        a["hits"] += r["correct"]
        a["base"] += r["baseline_correct"]
        a["total"] += 1
    return [{"year": y, "hits": a["hits"], "total": a["total"],
             "acc": a["hits"] / a["total"],
             "baseline_acc": a["base"] / a["total"]}
            for y, a in sorted(agg.items())]


def per_category_accuracy():
    agg = defaultdict(lambda: {"hits": 0, "total": 0, "base": 0})
    for r in races():
        a = agg[r["category"]]
        a["hits"] += r["correct"]
        a["base"] += r["baseline_correct"]
        a["total"] += 1
    out = []
    for cat, a in agg.items():
        out.append({"category": cat, "label": TONY_CATEGORIES[cat],
                    "group": CATEGORY_GROUP.get(cat, "?"),
                    "hits": a["hits"], "total": a["total"],
                    "acc": a["hits"] / a["total"],
                    "baseline_acc": a["base"] / a["total"]})
    out.sort(key=lambda d: d["acc"], reverse=True)
    return out


def per_group_accuracy():
    agg = defaultdict(lambda: {"hits": 0, "total": 0})
    for r in races():
        g = CATEGORY_GROUP.get(r["category"], "?")
        agg[g]["hits"] += r["correct"]
        agg[g]["total"] += 1
    return [{"group": g, "hits": a["hits"], "total": a["total"],
             "acc": a["hits"] / a["total"]}
            for g, a in sorted(agg.items())]


def calibration_bins(nbins=10):
    bins = defaultdict(lambda: {"hits": 0, "n": 0})
    for r in races():
        b = min(int(r["prob"] * nbins), nbins - 1)
        bins[b]["hits"] += r["correct"]
        bins[b]["n"] += 1
    out = []
    for b in range(nbins):
        if bins[b]["n"] == 0:
            continue
        lo, hi = b / nbins, (b + 1) / nbins
        out.append({"bin_lo": lo, "bin_hi": hi,
                    "predicted_mid": (lo + hi) / 2,
                    "actual": bins[b]["hits"] / bins[b]["n"],
                    "n": bins[b]["n"]})
    return out


def confidence_vs_correct():
    return [{"year": r["year"], "category": r["category"],
             "prob": r["prob"], "correct": r["correct"]} for r in races()]


def overall():
    rs = races()
    hits = sum(r["correct"] for r in rs)
    base = sum(r["baseline_correct"] for r in rs)
    return {"hits": hits, "total": len(rs),
            "acc": hits / len(rs), "baseline_acc": base / len(rs)}


def _tier(p):
    return "LOCK" if p > 0.70 else ("LEAN" if p > 0.45 else "TOSS-UP")


def prediction_confidences(year):
    """The model's confidence per category for a target year (e.g. 2026),
    trained on all other seasons. Used to chart the live forecast."""
    rby = _labeled_rows_by_year()
    target = next((s for s in load_all_seasons() if s["year"] == year), None)
    if target is None:
        return []
    train = [r for y, rows in rby.items() if y != year for r in rows]
    model = EnsembleTonyModel(weights=(0.5, 1.0, 1.0)).fit(train)
    bycat = defaultdict(list)
    for r in season_to_rows(target):
        bycat[r.category].append(r)
    out = []
    for cat in TONY_CATEGORIES:
        if cat not in bycat:
            continue
        top, prob, _ = model.predict_category(bycat[cat])[0]
        out.append({"category": cat, "label": TONY_CATEGORIES[cat],
                    "group": CATEGORY_GROUP.get(cat, "?"),
                    "pick": top.nominee_person or top.nominee_show,
                    "prob": prob, "tier": _tier(prob)})
    out.sort(key=lambda d: d["prob"], reverse=True)
    return out


if __name__ == "__main__":
    o = overall()
    print(f"Overall LOYO accuracy: {o['hits']}/{o['total']} = {o['acc']*100:.1f}% "
          f"(baseline {o['baseline_acc']*100:.1f}%)")
    print("\nPer-year:")
    for r in per_year_accuracy():
        print(f"  {r['year']}  {r['acc']*100:4.0f}%  ({r['hits']}/{r['total']})")
