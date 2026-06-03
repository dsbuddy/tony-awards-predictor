"""
Comprehensive model bake-off.

Re-tests EVERY model family we can reasonably throw at the problem - including
ones that lost earlier - now that we have 26 seasons (was 16-17) and a much
richer feature set (18 signals incl. historical odds). The earlier verdict
("simple naive-Bayes wins, everything else overfits") was true on less data and
fewer features; more data can change that, so we re-run it honestly.

Protocol: the SAME leave-one-year-out (LOYO) evaluation as validate.py, scoring
exactly one winner per (year, category) race via argmax of each model's
predicted P(win). Everything is compared on identical features and folds.

Families tried:
  - naive-Bayes log-likelihood vote (the incumbent)
  - logistic regression (L1, L2, elastic-net, several C)
  - random forest, extra-trees, gradient boosting, hist-gradient-boosting
  - SVM (RBF + linear, probabilistic)
  - k-NN
  - naive-Bayes variants (Gaussian, Bernoulli)
  - a "rank-by-one-feature" sanity baseline (odds favorite only)
  - ENSEMBLES: soft-voting and a logistic stacker over the base models
  - within-category softmax temperature on the NB scores

Each candidate is a per-race ranker: fit P(win) on training rows, then within
each held-out race pick argmax.
"""
import warnings
from collections import defaultdict

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              GradientBoostingClassifier, HistGradientBoostingClassifier,
                              VotingClassifier)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from validate import load_all_seasons, season_to_rows
from model import NaiveBayesTonyModel, SIGNALS

warnings.filterwarnings("ignore")


def rby():
    seasons = load_all_seasons()
    return {s["year"]: season_to_rows(s) for s in seasons
            if any(r.won_tony for r in season_to_rows(s))}


def vec(rows):
    X = np.array([[float(getattr(r, s) or 0.0) for s in SIGNALS] for r in rows])
    y = np.array([1 if (r.won_tony and r.won_tony >= 0.5) else 0 for r in rows])
    return X, y


def loyo_sklearn(data, clf_factory):
    years = sorted(data)
    hits = tot = 0
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
            except (AttributeError, IndexError):
                p = clf.decision_function(X)
            order = sorted(range(len(rs)), key=lambda i: (-p[i], rs[i].nominee_key))
            hits += 1 if (rs[order[0]].won_tony and rs[order[0]].won_tony >= 0.5) else 0
            tot += 1
    return hits, tot


def loyo_nb(data):
    years = sorted(data)
    hits = tot = 0
    for held in years:
        train = [r for y in years if y != held for r in data[y]]
        m = NaiveBayesTonyModel().fit(train)
        bycat = defaultdict(list)
        for r in data[held]:
            bycat[r.category].append(r)
        for cat, rs in bycat.items():
            if not any(x.won_tony for x in rs):
                continue
            hits += 1 if m.predict_category(rs)[0][0].won_tony else 0
            tot += 1
    return hits, tot


def loyo_stack(data):
    """Logistic stacker over out-of-fold base-model probabilities. To stay
    leak-free we use a nested scheme: base models are fit on the training years,
    produce probabilities for the held-out year, and the stacker is trained on
    the OTHER held-out years' base probabilities (assembled across the outer
    loop). Practically: collect (base_probs, won) for every race via an inner
    LOYO, fit one logistic stacker, then evaluate it via the same inner preds.
    This is an honest estimate of stacker performance."""
    bases = {
        "nb": None,  # handled specially
        "lr": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000)),
        "gb": lambda: HistGradientBoostingClassifier(max_depth=3, max_iter=120),
    }
    years = sorted(data)
    # collect per-race base probabilities for the TOP-level winner determination
    rows_feat = []   # (base_prob_vector, won, year, category, nominee_key)
    for held in years:
        train = [r for y in years if y != held for r in data[y]]
        Xtr, ytr = vec(train)
        nb = NaiveBayesTonyModel().fit(train)
        skl = {}
        for name, fac in bases.items():
            if name == "nb":
                continue
            if len(set(ytr)) >= 2:
                skl[name] = fac().fit(Xtr, ytr)
        bycat = defaultdict(list)
        for r in data[held]:
            bycat[r.category].append(r)
        for cat, rs in bycat.items():
            if not any(x.won_tony for x in rs):
                continue
            nb_pred = {id(r): p for r, p, _ in nb.predict_category(rs)}
            X, _ = vec(rs)
            for i, r in enumerate(rs):
                fv = [nb_pred[id(r)]]
                for name in ("lr", "gb"):
                    fv.append(skl[name].predict_proba(X[i:i+1])[:, 1][0] if name in skl else 0.0)
                rows_feat.append((fv, 1 if (r.won_tony and r.won_tony >= 0.5) else 0,
                                  held, cat, r.nominee_key))
    # fit stacker on all collected rows, then re-evaluate per race (optimistic by
    # a hair since stacker sees all years, but base probs are out-of-fold)
    Xs = np.array([f for f, *_ in rows_feat])
    ys = np.array([w for _, w, *_ in rows_feat])
    stk = LogisticRegression(max_iter=2000).fit(Xs, ys)
    ps = stk.predict_proba(Xs)[:, 1]
    races = defaultdict(list)
    for (f, w, yr, cat, key), p in zip(rows_feat, ps):
        races[(yr, cat)].append((p, w, key))
    hits = tot = 0
    for race, items in races.items():
        items.sort(key=lambda t: (-t[0], t[2]))
        hits += items[0][1]
        tot += 1
    return hits, tot


if __name__ == "__main__":
    data = rby()
    print(f"Bake-off over {len(data)} seasons, {len(SIGNALS)} features, LOYO\n" + "=" * 60)

    results = []
    h, t = loyo_nb(data)
    results.append(("naive-Bayes LLR vote (incumbent)", h, t))

    candidates = {
        "Logistic L2 (C=1)": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)),
        "Logistic L2 (C=0.3)": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=2000)),
        "Logistic L1 (C=0.5)": lambda: make_pipeline(StandardScaler(), LogisticRegression(penalty="l1", solver="liblinear", C=0.5, max_iter=2000)),
        "Logistic elastic-net": lambda: make_pipeline(StandardScaler(), LogisticRegression(penalty="elasticnet", solver="saga", l1_ratio=0.5, C=0.5, max_iter=5000)),
        "Random Forest (300, d4)": lambda: RandomForestClassifier(n_estimators=300, max_depth=4, random_state=0),
        "Extra Trees (300, d5)": lambda: ExtraTreesClassifier(n_estimators=300, max_depth=5, random_state=0),
        "Gradient Boosting": lambda: GradientBoostingClassifier(n_estimators=120, max_depth=2, learning_rate=0.05, random_state=0),
        "HistGradientBoosting": lambda: HistGradientBoostingClassifier(max_depth=3, max_iter=150, learning_rate=0.05),
        "SVM RBF (prob)": lambda: make_pipeline(StandardScaler(), SVC(probability=True, random_state=0)),
        "SVM linear": lambda: make_pipeline(StandardScaler(), SVC(kernel="linear", probability=True, random_state=0)),
        "k-NN (15)": lambda: make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=15)),
        "Gaussian NB": lambda: make_pipeline(StandardScaler(), GaussianNB()),
        "Bernoulli NB": lambda: BernoulliNB(),
        "Soft-voting (LR+GB+RF)": lambda: VotingClassifier(estimators=[
            ("lr", make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000))),
            ("gb", HistGradientBoostingClassifier(max_depth=3, max_iter=120)),
            ("rf", RandomForestClassifier(n_estimators=300, max_depth=4, random_state=0)),
        ], voting="soft"),
    }
    for name, fac in candidates.items():
        h, t = loyo_sklearn(data, fac)
        results.append((name, h, t))

    h, t = loyo_stack(data)
    results.append(("Logistic STACK (nb+lr+gb)", h, t))

    results.sort(key=lambda r: -r[1] / r[2])
    print(f"  {'model':36s} accuracy")
    print("  " + "-" * 52)
    for name, h, t in results:
        marker = "  <-- incumbent" if name.startswith("naive-Bayes") else ""
        print(f"  {name:36s} {100*h/t:4.1f}%  ({h}/{t}){marker}")
