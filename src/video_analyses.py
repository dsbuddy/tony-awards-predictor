"""
Extra analyses behind the expanded video chart set. Each function computes a
REAL result from the data (not decoration) via the same leave-one-year-out
discipline where prediction is involved. video_figures2.py renders these.
"""
from collections import defaultdict

from validate import load_all_seasons, season_to_rows, baseline_pick
from model import NaiveBayesTonyModel, SIGNALS, CATEGORY_GROUP
from model_ensemble import EnsembleTonyModel
import model as M


def _rby():
    return {s["year"]: season_to_rows(s) for s in load_all_seasons()
            if any(r.won_tony for r in season_to_rows(s))}


def _winner(rs):
    return next((r for r in rs if r.won_tony and r.won_tony >= 0.5), None)


# --- 1. Single-precursor power ranking: if you ONLY had one award, how often
#        does its category winner match the Tony winner? (the "kingmaker") ---
def precursor_power():
    rby = _rby()
    awards = {"Drama Desk": "dd_won", "Outer Critics": "occ_won",
              "Drama League": "dl_won", "Critics' Circle": "nydcc_won",
              "Pundit favorite": "odds_favorite"}
    out = {}
    for label, attr in awards.items():
        hit = tot = 0
        for rows in rby.values():
            bycat = defaultdict(list)
            for r in rows:
                bycat[r.category].append(r)
            for cat, rs in bycat.items():
                win = _winner(rs)
                if not win:
                    continue
                # nominees this award flagged
                flagged = [r for r in rs if getattr(r, attr, 0) and getattr(r, attr) >= 0.5]
                if not flagged:
                    continue  # this award didn't weigh in on this race
                tot += 1
                if any(r is win for r in flagged):
                    hit += 1
        out[label] = (hit, tot, hit / tot if tot else 0)
    return out


# --- 2. The sweep effect: when ONE nominee wins both Drama Desk AND Outer
#        Critics in a category, how often do they win the Tony? ---
def sweep_effect():
    rby = _rby()
    buckets = {"won both DD + OCC": [0, 0], "won just one": [0, 0], "won neither": [0, 0]}
    for rows in rby.values():
        bycat = defaultdict(list)
        for r in rows:
            bycat[r.category].append(r)
        for cat, rs in bycat.items():
            win = _winner(rs)
            if not win:
                continue
            dd, occ = bool(win.dd_won), bool(win.occ_won)
            # only meaningful when both awards actually ran this category for someone
            ran_dd = any(r.dd_won or r.dd_nominated for r in rs)
            ran_occ = any(r.occ_won or r.occ_nominated for r in rs)
            if not (ran_dd and ran_occ):
                continue
            key = "won both DD + OCC" if (dd and occ) else ("won just one" if (dd or occ) else "won neither")
            buckets[key][1] += 1
            buckets[key][0] += 1  # the winner is by definition the winner; we want P(Tony win | precursor status)
    # The above counts winners; we actually want, among the NOMINEE who had each
    # precursor status, the rate they won. Recompute properly:
    buckets = {"won both DD + OCC": [0, 0], "won just one": [0, 0], "won neither": [0, 0]}
    for rows in rby.values():
        bycat = defaultdict(list)
        for r in rows:
            bycat[r.category].append(r)
        for cat, rs in bycat.items():
            if not _winner(rs):
                continue
            ran_dd = any(r.dd_won or r.dd_nominated for r in rs)
            ran_occ = any(r.occ_won or r.occ_nominated for r in rs)
            if not (ran_dd and ran_occ):
                continue
            for r in rs:
                dd, occ = bool(r.dd_won), bool(r.occ_won)
                key = "won both DD + OCC" if (dd and occ) else ("won just one" if (dd or occ) else "won neither")
                buckets[key][1] += 1
                if r.won_tony and r.won_tony >= 0.5:
                    buckets[key][0] += 1
    return {k: (v[0], v[1], v[0] / v[1] if v[1] else 0) for k, v in buckets.items()}


# --- 3. Agreement -> accuracy: the more precursors that agree on one nominee,
#        the more likely the model (and reality) lands there ---
def agreement_accuracy():
    rby = _rby()
    buckets = defaultdict(lambda: [0, 0])  # n_agreeing -> [tony_wins_for_consensus, races]
    for rows in rby.values():
        bycat = defaultdict(list)
        for r in rows:
            bycat[r.category].append(r)
        for cat, rs in bycat.items():
            win = _winner(rs)
            if not win:
                continue
            # how many precursor bodies the eventual WINNER had won. This asks:
            # "when N bodies backed the winner, ...", but to measure PREDICTIVE
            # value we instead ask, of the nominee the precursors most agreed on,
            # how often they won - skipping races where no nominee won any
            # precursor (no consensus to test).
            best = max(rs, key=lambda r: r.n_precursor_wins)
            if best.n_precursor_wins == 0:
                continue  # no consensus pick in this race; nothing to measure
            n_agree = min(int(best.n_precursor_wins), 4)
            buckets[n_agree][1] += 1
            if best is win:
                buckets[n_agree][0] += 1
    return {k: (v[0], v[1], v[0] / v[1] if v[1] else 0) for k, v in sorted(buckets.items())}


# --- 4. Model vs pundits head-to-head (LOYO model vs just-follow-Gold-Derby) ---
def model_vs_pundits():
    rby = _rby()
    years = sorted(rby)
    model_hit = model_tot = pundit_hit = pundit_tot = 0
    for held in years:
        train = [r for y in years if y != held for r in rby[y]]
        mdl = EnsembleTonyModel(weights=(0.5, 1.0, 1.0)).fit(train)
        bycat = defaultdict(list)
        for r in rby[held]:
            bycat[r.category].append(r)
        for cat, rs in bycat.items():
            win = _winner(rs)
            if not win:
                continue
            top = mdl.predict_category(rs)[0][0]
            model_tot += 1
            model_hit += 1 if top is win else 0
            favs = [r for r in rs if getattr(r, "odds_favorite", 0) >= 0.5]
            if favs:
                pundit_tot += 1
                pundit_hit += 1 if any(r is win for r in favs) else 0
    return {"model": (model_hit, model_tot, model_hit / model_tot),
            "pundits": (pundit_hit, pundit_tot, pundit_hit / pundit_tot if pundit_tot else 0)}


# --- 5. The build-up ablation (accuracy as each feature group is added) ---
def ablation_climb():
    rby = _rby()
    P = ["dd_won", "dd_nominated", "occ_won", "occ_nominated", "dl_won", "dl_nominated", "nydcc_won"]
    FR = ["is_frontrunner"]
    EX = ["ext_pundit_fav", "ext_rave", "ext_wellreviewed", "ext_hit", "ext_flop", "ext_open", "ext_closed"]
    OD = ["odds_favorite", "odds_strong_fav"]
    stages = [("Precursors", P), ("+ Sweep", P + FR),
              ("+ Reviews/\ncommercial", P + FR + EX), ("+ Pundit\nodds", P + FR + EX + OD)]
    years = sorted(rby)
    out = []
    for name, sigs in stages:
        orig = M.SIGNALS[:]
        M.SIGNALS[:] = sigs
        hit = tot = 0
        try:
            for held in years:
                train = [r for y in years if y != held for r in rby[y]]
                m = NaiveBayesTonyModel().fit(train)
                bycat = defaultdict(list)
                for r in rby[held]:
                    bycat[r.category].append(r)
                for cat, rs in bycat.items():
                    if not _winner(rs):
                        continue
                    hit += 1 if m.predict_category(rs)[0][0].won_tony else 0
                    tot += 1
        finally:
            M.SIGNALS[:] = orig
        out.append((name, hit / tot))
    return out


# --- 6. Group accuracy (production / acting / creative / design) ---
def group_accuracy():
    import analytics
    return {g["group"]: g["acc"] for g in analytics.per_group_accuracy()}


# --- 7. Best Play, every year: did the model nail it? (the "solved" category) ---
def category_track_record(cat="best_play"):
    rby = _rby()
    years = sorted(rby)
    out = []
    for held in years:
        rows = [r for r in rby[held] if r.category == cat]
        if not rows or not _winner(rows):
            continue
        train = [r for y in years if y != held for r in rby[y]]
        m = EnsembleTonyModel(weights=(0.5, 1.0, 1.0)).fit(train)
        top = m.predict_category(rows)[0][0]
        out.append((held, bool(top.won_tony and top.won_tony >= 0.5)))
    return out


if __name__ == "__main__":
    print("PRECURSOR POWER:"); [print(f"  {k}: {v[2]*100:.0f}% ({v[0]}/{v[1]})") for k, v in precursor_power().items()]
    print("\nSWEEP EFFECT:"); [print(f"  {k}: {v[2]*100:.0f}% win the Tony ({v[0]}/{v[1]})") for k, v in sweep_effect().items()]
    print("\nAGREEMENT -> ACCURACY:"); [print(f"  {k} bodies agree: {v[2]*100:.0f}% ({v[0]}/{v[1]})") for k, v in agreement_accuracy().items()]
    print("\nMODEL vs PUNDITS:"); [print(f"  {k}: {v[2]*100:.0f}% ({v[0]}/{v[1]})") for k, v in model_vs_pundits().items()]
    print("\nABLATION CLIMB:"); [print(f"  {n.replace(chr(10),' ')}: {a*100:.0f}%") for n, a in ablation_climb()]
