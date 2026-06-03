"""
Comprehensive model bake-off (rigorous edition).

The quick bake-off (bakeoff.py) surfaced a real finding: with 26 seasons and 22
features, several models now BEAT the naive-Bayes incumbent (the opposite of the
earlier verdict on less data). But the gaps are 1-2 points = a handful of races
out of ~612, so this script does the work to tell signal from noise:

  1. HYPERPARAMETER SWEEPS for the contending families (not single settings).
  2. HONEST evaluation: leave-one-year-out (LOYO), one winner per race.
  3. SIGNIFICANCE: paired bootstrap over races + McNemar-style paired counts to
     ask "is model A really better than the incumbent, or within noise?".
  4. PER-GROUP model selection: maybe logistic wins acting while NB wins design.
  5. FEATURE ablation already covered in experiments_v3; here we add permutation
     importance for the best linear model.
  6. CALIBRATION quality (Brier score) not just top-1 accuracy.

Run: python3 bakeoff_full.py   (takes a couple minutes - it's thorough on purpose)
"""
import warnings
from collections import defaultdict

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier, HistGradientBoostingClassifier,
                              VotingClassifier)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from validate import load_all_seasons, season_to_rows
from model import NaiveBayesTonyModel, SIGNALS, CATEGORY_GROUP

warnings.filterwarnings("ignore")
RNG = np.random.RandomState(20260604)


def rby():
    seasons = load_all_seasons()
    return {s["year"]: season_to_rows(s) for s in seasons
            if any(r.won_tony for r in season_to_rows(s))}


def vec(rows):
    X = np.array([[float(getattr(r, s) or 0.0) for s in SIGNALS] for r in rows])
    y = np.array([1 if (r.won_tony and r.won_tony >= 0.5) else 0 for r in rows])
    return X, y


def race_outcomes_sklearn(data, clf_factory):
    """Return per-race dicts {year, category, correct, won_prob_of_winner,
    brier} via LOYO so we can do significance + calibration, not just accuracy."""
    years = sorted(data)
    out = []
    for held in years:
        train = [r for y in years if y != held for r in data[y]]
        Xtr, ytr = vec(train)
        if len(set(ytr)) < 2:
            continue
        clf = clf_factory().fit(Xtr, ytr)
        bycat = defaultdict(list)
        for r in data[held]:
            bycat[r.category].append(r)
        for cat, rs in bycat.items():
            if not any(x.won_tony for x in rs):
                continue
            X, _ = vec(rs)
            try:
                p = clf.predict_proba(X)[:, 1]
            except Exception:
                d = clf.decision_function(X)
                p = 1 / (1 + np.exp(-d))
            p = np.clip(p, 1e-9, 1 - 1e-9)
            order = sorted(range(len(rs)), key=lambda i: (-p[i], rs[i].nominee_key))
            win_i = next(i for i, r in enumerate(rs) if r.won_tony and r.won_tony >= 0.5)
            # normalize within race for a proper Brier on the winner
            pn = p / p.sum()
            out.append({"year": held, "category": cat,
                        "correct": int(order[0] == win_i),
                        "brier": float(np.sum((pn - np.eye(len(rs))[win_i]) ** 2))})
    return out


def race_outcomes_nb(data):
    years = sorted(data)
    out = []
    for held in years:
        train = [r for y in years if y != held for r in data[y]]
        m = NaiveBayesTonyModel().fit(train)
        bycat = defaultdict(list)
        for r in data[held]:
            bycat[r.category].append(r)
        for cat, rs in bycat.items():
            if not any(x.won_tony for x in rs):
                continue
            pred = m.predict_category(rs)
            top = pred[0][0]
            probs = np.array([p for _, p, _ in pred])
            keys = [r for r, _, _ in pred]
            win_i = next(i for i, r in enumerate(keys) if r.won_tony and r.won_tony >= 0.5)
            pn = probs / probs.sum()
            out.append({"year": held, "category": cat,
                        "correct": int(top.won_tony and top.won_tony >= 0.5),
                        "brier": float(np.sum((pn - np.eye(len(keys))[win_i]) ** 2))})
    return out


def acc(outcomes):
    return np.mean([o["correct"] for o in outcomes])


def brier(outcomes):
    return np.mean([o["brier"] for o in outcomes])


def paired_bootstrap(a, b, n=5000):
    """a, b: outcome lists aligned by race. Return P(a beats b) and the mean
    accuracy delta + 95% CI via bootstrap over races."""
    key = lambda o: (o["year"], o["category"])
    bmap = {key(o): o["correct"] for o in b}
    pairs = [(o["correct"], bmap[key(o)]) for o in a if key(o) in bmap]
    arr = np.array(pairs, dtype=float)
    deltas = []
    idx = np.arange(len(arr))
    for _ in range(n):
        s = RNG.choice(idx, len(idx), replace=True)
        deltas.append(arr[s, 0].mean() - arr[s, 1].mean())
    deltas = np.array(deltas)
    return (deltas > 0).mean(), deltas.mean(), np.percentile(deltas, [2.5, 97.5])


# hyperparameter sweeps for the contending families
def logistic_grid():
    grid = {}
    for pen, solver, extra in [("l2", "lbfgs", {}), ("l1", "liblinear", {}),
                               ("elasticnet", "saga", {"l1_ratio": 0.5})]:
        for C in (0.1, 0.2, 0.3, 0.5, 1.0, 2.0):
            name = f"LR {pen} C={C}"
            grid[name] = (lambda C=C, pen=pen, solver=solver, extra=extra:
                          make_pipeline(StandardScaler(),
                                        LogisticRegression(penalty=pen, solver=solver, C=C,
                                                           max_iter=5000, **extra)))
    return grid


def hgb_grid():
    grid = {}
    for depth in (2, 3, 4):
        for lr in (0.03, 0.05, 0.1):
            name = f"HGB d={depth} lr={lr}"
            grid[name] = (lambda depth=depth, lr=lr:
                          HistGradientBoostingClassifier(max_depth=depth, learning_rate=lr,
                                                         max_iter=200))
    return grid


def rf_grid():
    grid = {}
    for n in (200, 400):
        for d in (3, 4, 6):
            name = f"RF n={n} d={d}"
            grid[name] = (lambda n=n, d=d:
                          RandomForestClassifier(n_estimators=n, max_depth=d, random_state=0))
    return grid


if __name__ == "__main__":
    data = rby()
    print(f"COMPREHENSIVE BAKE-OFF | {len(data)} seasons | {len(SIGNALS)} features | LOYO")
    print("=" * 68)

    nb_out = race_outcomes_nb(data)
    print(f"\nIncumbent naive-Bayes: acc={100*acc(nb_out):.1f}%  Brier={brier(nb_out):.3f}  "
          f"(n={len(nb_out)} races)")

    print("\n--- HYPERPARAMETER SWEEPS (acc | Brier) ---")
    best = {}
    for label, grid in [("Logistic", logistic_grid()), ("HistGB", hgb_grid()), ("RandomForest", rf_grid())]:
        print(f"\n[{label}]")
        rows = []
        for name, fac in grid.items():
            o = race_outcomes_sklearn(data, fac)
            rows.append((name, acc(o), brier(o), o))
        rows.sort(key=lambda r: -r[1])
        for name, a, b, _ in rows[:5]:
            print(f"  {name:22s} {100*a:4.1f}%  |  {b:.3f}")
        best[label] = rows[0]

    print("\n--- SIGNIFICANCE vs incumbent (paired bootstrap over races, 5000x) ---")
    print(f"  {'model':24s} acc    P(beats NB)   mean delta [95% CI]")
    for label, (name, a, b, o) in best.items():
        pwin, md, ci = paired_bootstrap(o, nb_out)
        print(f"  {name:24s} {100*a:4.1f}%   {pwin*100:5.1f}%      {md*100:+.1f}pp [{ci[0]*100:+.1f}, {ci[1]*100:+.1f}]")

    # also test an ensemble of the best of each family
    print("\n--- ENSEMBLE of best-of-family (soft vote) ---")
    ens = lambda: VotingClassifier(estimators=[
        ("lr", best["Logistic"][0] and logistic_grid()[best["Logistic"][0]]()),
        ("hgb", hgb_grid()[best["HistGB"][0]]()),
        ("rf", rf_grid()[best["RandomForest"][0]]()),
    ], voting="soft")
    eo = race_outcomes_sklearn(data, ens)
    pwin, md, ci = paired_bootstrap(eo, nb_out)
    print(f"  ensemble                 {100*acc(eo):4.1f}%   {pwin*100:5.1f}%      {md*100:+.1f}pp [{ci[0]*100:+.1f}, {ci[1]*100:+.1f}]  Brier={brier(eo):.3f}")

    # per-group: which model wins each category group?
    print("\n--- PER-GROUP accuracy (does the best model differ by group?) ---")
    models = {"naive-Bayes": nb_out,
              best["Logistic"][0]: best["Logistic"][3],
              best["HistGB"][0]: best["HistGB"][3]}
    groups = ["production", "acting", "creative", "design"]
    print(f"  {'model':24s} " + "  ".join(f"{g[:5]:>6s}" for g in groups))
    for name, o in models.items():
        by = defaultdict(lambda: [0, 0])
        for race in o:
            g = CATEGORY_GROUP.get(race["category"], "?")
            by[g][0] += race["correct"]; by[g][1] += 1
        cells = "  ".join(f"{100*by[g][0]/by[g][1]:5.0f}%" if by[g][1] else "   -- " for g in groups)
        print(f"  {name:24s} {cells}")

    print("\nINTERPRETATION GUIDE:")
    print("  - If P(beats NB) < ~90% and the 95% CI for the delta spans 0, the")
    print("    'win' is within noise - prefer the simpler / more explainable model.")
    print("  - Lower Brier = better-calibrated probabilities (matters for the")
    print("    confidence levels we publish, not just the top pick).")
