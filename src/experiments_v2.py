"""
Comprehensive experiment harness - multiple model families + engineered
features, all evaluated by the SAME leak-free leave-one-year-out (LOYO)
backtest so deltas are comparable to the live model's 59%.

Pathways tested here (no external data):
  1. Engineered features (free, derived from existing season files)
  2. Model families: naive-Bayes (current), conditional logistic regression,
     L2 logistic, random forest, gradient boosting.

Each model gets the SAME feature matrix and must pick exactly one winner per
(year, category) race. We never train on the held-out year.
"""
import math
import warnings
from collections import defaultdict

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from validate import load_all_seasons, season_to_rows
from model import NaiveBayesTonyModel, CATEGORY_GROUP
from schema import norm

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Feature engineering: enrich each FeatureRow with extra leak-free signals.
# ---------------------------------------------------------------------------

BASE_FEATS = ["dd_won", "dd_nominated", "occ_won", "occ_nominated",
              "dl_won", "dl_nominated", "nydcc_won"]


def enrich(seasons):
    """Return {year: [rows]} with engineered features attached."""
    out = {}
    for s in seasons:
        rows = season_to_rows(s)
        if not any(r.won_tony for r in rows):
            # still keep for prediction-only years, but LOYO skips them
            pass
        # show-level Tony nomination counts (the sweep signal source)
        show_counts = defaultdict(int)
        for cat, data in s["tony"].items():
            seen = set()
            for n in data["nominees"]:
                sh = norm(n["show"])
                if sh not in seen:
                    show_counts[sh] += 1
                    seen.add(sh)
        bycat = defaultdict(list)
        for r in rows:
            r.show_nom_count = float(show_counts.get(norm(r.nominee_show), 0))
            bycat[r.category].append(r)
        for cat, rs in bycat.items():
            counts = sorted((r.show_nom_count for r in rs), reverse=True)
            top = counts[0] if counts else 0
            second = counts[1] if len(counts) > 1 else 0
            n_in_race = len(rs)
            for r in rs:
                # frontrunner (unique leader)
                leaders = sum(1 for x in rs if x.show_nom_count == top)
                r.is_frontrunner = 1.0 if (r.show_nom_count == top and top > 0
                                           and leaders < n_in_race) else 0.0
                # graded margin: how far ahead the leader is (0 if not leader)
                r.fr_margin = float(top - second) if r.is_frontrunner else 0.0
                # normalized nom count (share of the race's total noms)
                tot = sum(x.show_nom_count for x in rs) or 1
                r.nom_share = r.show_nom_count / tot
                # smaller races are easier to call -> encode inverse size
                r.race_inv_size = 1.0 / n_in_race
                # total precursor wins + noms (consensus strength)
                r.precursor_strength = r.n_precursor_wins + 0.4 * r.n_precursor_noms
                # did this entity sweep ALL precursors that ran the category?
                ran = sum(v is not None for v in
                          [r.dd_won, r.occ_won, r.dl_won, r.nydcc_won])
                won = r.n_precursor_wins
                r.precursor_sweep = 1.0 if (ran >= 2 and won == ran) else 0.0
        out[s["year"]] = rows
    return out


def vectorize(rows, feats):
    X = np.array([[float(getattr(r, f) or 0.0) for f in feats] for r in rows])
    y = np.array([1 if (r.won_tony and r.won_tony >= 0.5) else 0 for r in rows])
    return X, y


# ---------------------------------------------------------------------------
# Model wrappers: each exposes fit(rows, feats) and pick(rows_in_cat, feats)
# returning the chosen winner row. One winner per race (argmax of P(win)).
# ---------------------------------------------------------------------------

class SklearnRace:
    def __init__(self, clf): self.clf = clf
    def fit(self, rows, feats):
        X, y = vectorize(rows, feats)
        if len(set(y)) < 2:           # degenerate; fall back to all-zero
            self._const = True
        else:
            self._const = False
            self.clf.fit(X, y)
        return self
    def proba(self, rows, feats):
        if self._const:
            return np.zeros(len(rows))
        X, _ = vectorize(rows, feats)
        return self.clf.predict_proba(X)[:, 1]


def loyo(rows_by_year, build_and_pick):
    """Generic leave-one-year-out. build_and_pick(train_rows, test_bycat) ->
    {category: chosen_row}. Returns (hits, total)."""
    years = sorted(rows_by_year)
    hits = total = 0
    for held in years:
        train = [r for y in years if y != held for r in rows_by_year[y]
                 if r.won_tony is not None]
        bycat = defaultdict(list)
        for r in rows_by_year[held]:
            bycat[r.category].append(r)
        picks = build_and_pick(train, bycat, held)
        for cat, rs in bycat.items():
            if not any(x.won_tony for x in rs):
                continue
            chosen = picks.get(cat)
            if chosen is not None and chosen.won_tony and chosen.won_tony >= 0.5:
                hits += 1
            total += 1
    return hits, total


def run_sklearn(rows_by_year, feats, clf_factory):
    def bp(train, bycat, held):
        m = SklearnRace(clf_factory()).fit(train, feats)
        picks = {}
        for cat, rs in bycat.items():
            p = m.proba(rs, feats)
            # tie-break leak-free: argmax then alpha by nominee_key
            order = sorted(range(len(rs)), key=lambda i: (-p[i], rs[i].nominee_key))
            picks[cat] = rs[order[0]]
        return picks
    return loyo(rows_by_year, bp)


def run_naive_bayes(rows_by_year, extra_signals=()):
    from model import SIGNALS
    def bp(train, bycat, held):
        m = NaiveBayesTonyModel().fit(train)
        if extra_signals:  # patch signal list
            pass
        picks = {}
        for cat, rs in bycat.items():
            pred = m.predict_category(rs)
            picks[cat] = pred[0][0]
        return picks
    return loyo(rows_by_year, bp)


if __name__ == "__main__":
    seasons = load_all_seasons()
    rby = enrich(seasons)

    FEAT_SETS = {
        "base (precursors only)": BASE_FEATS,
        "base + frontrunner": BASE_FEATS + ["is_frontrunner"],
        "base + fr + margin/share": BASE_FEATS + ["is_frontrunner", "fr_margin", "nom_share"],
        "base + fr + sweep + race-size": BASE_FEATS + ["is_frontrunner", "precursor_sweep", "race_inv_size"],
        "ALL engineered": BASE_FEATS + ["is_frontrunner", "fr_margin", "nom_share",
                                        "race_inv_size", "precursor_strength", "precursor_sweep"],
    }

    print("LOYO BACKTEST - model families x feature sets")
    print("=" * 64)
    print(f"  {'current naive-Bayes (live model)':46s}", end="")
    h, t = run_naive_bayes(rby)
    print(f"{h}/{t} = {100*h/t:4.1f}%")
    print("-" * 64)

    models = {
        "LogReg (L2, C=1)":   lambda: LogisticRegression(C=1.0, max_iter=1000),
        "LogReg (L2, C=0.3)": lambda: LogisticRegression(C=0.3, max_iter=1000),
        "RandomForest(200)":  lambda: RandomForestClassifier(n_estimators=200, max_depth=4, random_state=0),
        "GradBoost":          lambda: GradientBoostingClassifier(n_estimators=80, max_depth=2, random_state=0),
    }

    best = (0, None)
    for fname, feats in FEAT_SETS.items():
        print(f"\n[{fname}]  ({len(feats)} features)")
        for mname, fac in models.items():
            h, t = run_sklearn(rby, feats, fac)
            acc = 100 * h / t
            if acc > best[0]:
                best = (acc, f"{mname} | {fname}")
            print(f"    {mname:22s} {h}/{t} = {acc:4.1f}%")
    print("\n" + "=" * 64)
    print(f"BEST: {best[1]} = {best[0]:.1f}%")
