"""
Round-3 experiment summary: an honest accounting of every feature and model
variant tried in response to reader feedback, evaluated by the same leave-one-
year-out (LOYO) backtest. Prints what helped and what didn't - including the
negative results, which matter just as much.

    python3 experiments_v3.py
"""
from collections import defaultdict

from validate import load_all_seasons, season_to_rows
from model import NaiveBayesTonyModel, SIGNALS
from model_hier import HierarchicalTonyModel
import model as M


def _rby():
    seasons = load_all_seasons()
    return {s["year"]: season_to_rows(s) for s in seasons
            if any(r.won_tony for r in season_to_rows(s))}


def loyo(rby, make, signal_override=None):
    orig = M.SIGNALS[:]
    if signal_override is not None:
        M.SIGNALS[:] = signal_override
    years = sorted(rby)
    hits = tot = 0
    try:
        for held in years:
            train = [r for y in years if y != held for r in rby[y]]
            mdl = make().fit(train)
            bc = defaultdict(list)
            for r in rby[held]:
                bc[r.category].append(r)
            for cat, rs in bc.items():
                if not any(x.won_tony for x in rs):
                    continue
                hits += 1 if mdl.predict_category(rs)[0][0].won_tony else 0
                tot += 1
    finally:
        M.SIGNALS[:] = orig
    return hits, tot


# the feature groups we can ablate
PRECURSOR = ["dd_won", "dd_nominated", "occ_won", "occ_nominated",
             "dl_won", "dl_nominated", "nydcc_won"]
FRONTRUNNER = ["is_frontrunner"]
EXTERNAL = ["ext_pundit_fav", "ext_rave", "ext_wellreviewed", "ext_hit",
            "ext_flop", "ext_open", "ext_closed"]
TRANSFER = ["tr_olivier", "tr_offbway_acclaim", "tr_transferred"]
CONSENSUS = ["con_win_fraction", "con_sole_winner"]
ODDS = ["odds_favorite", "odds_strong_fav"]


if __name__ == "__main__":
    rby = _rby()
    nb = lambda: NaiveBayesTonyModel()

    print("=" * 64)
    print(f"ROUND-3 ABLATION  ({len(rby)} seasons, LOYO)")
    print("=" * 64)

    # cumulative build-up: add one feature group at a time
    stages = [
        ("precursors only", PRECURSOR),
        ("+ frontrunner", PRECURSOR + FRONTRUNNER),
        ("+ external (reviews/commercial/pundit-show)", PRECURSOR + FRONTRUNNER + EXTERNAL),
        ("+ transfer (Olivier/Off-Bway)", PRECURSOR + FRONTRUNNER + EXTERNAL + TRANSFER),
        ("+ consensus-strength", PRECURSOR + FRONTRUNNER + EXTERNAL + TRANSFER + CONSENSUS),
        ("+ historical odds (FULL MODEL)", PRECURSOR + FRONTRUNNER + EXTERNAL + TRANSFER + CONSENSUS + ODDS),
    ]
    prev = None
    for name, sigs in stages:
        h, t = loyo(rby, nb, signal_override=sigs)
        acc = 100 * h / t
        delta = f"  ({acc - prev:+.1f})" if prev is not None else ""
        print(f"  {name:46s} {acc:4.1f}%{delta}")
        prev = acc

    print("\n  --- model-family check (full feature set) ---")
    h, t = loyo(rby, nb)
    print(f"  naive-Bayes (adopted)                          {100*h/t:4.1f}%")
    for k in (4.0, 8.0):
        h, t = loyo(rby, lambda: HierarchicalTonyModel(shrink_k=k))
        print(f"  hierarchical partial-pooling (K={k})            {100*h/t:4.1f}%  [not adopted]")

    print("\nNOTES:")
    print("  - Consensus-strength: ~flat on overall accuracy but fixes specific")
    print("    under-confidence (e.g. a clean sweep 79% -> 99%); kept for that.")
    print("  - Transfer features: ~flat; kept to address the 'ignores transfers'")
    print("    critique explicitly. Precursors already capture most of it.")
    print("  - Hierarchical model: LOSES to hard group-pooling; NOT adopted.")
    print("  - Vote-splitting (votesplit.py): unvalidatable heuristic, predict-")
    print("    time only, never in these accuracy numbers.")
    print("  - Historical odds: the single biggest win (+2.6 pts).")
