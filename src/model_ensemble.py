"""
Ensemble model: combine the naive-Bayes vote with a regularized logistic model
(and optionally gradient boosting), averaging their within-race win
probabilities.

Rationale: the comprehensive bake-off (bakeoff_full.py) showed logistic
elastic-net and HistGB modestly beat the naive-Bayes incumbent on accuracy and
clearly beat it on calibration (Brier 0.48 vs 0.55), but the accuracy gain's
95% CI still spanned 0. An ensemble that BLENDS them can be more robust than any
single model when they make different mistakes - the NB captures the
hand-reasoned signal structure, the logistic captures linear feature weights
with regularization, and averaging cancels idiosyncratic errors.

This module exposes the same predict_category() interface as NaiveBayesTonyModel
so it drops into predict.py / validate.py / analytics.py unchanged. Whether it
actually beats the single best model is tested in __main__ and reported
honestly - we only recommend adopting it if it wins or ties on accuracy AND
improves calibration.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from model import NaiveBayesTonyModel, SIGNALS


def _vec(rows):
    return np.array([[float(getattr(r, s) or 0.0) for s in SIGNALS] for r in rows])


def _vecy(rows):
    return (_vec(rows),
            np.array([1 if (r.won_tony and r.won_tony >= 0.5) else 0 for r in rows]))


class EnsembleTonyModel:
    """Soft-average of NB + logistic (+ optional HGB) within-race probabilities.

    weights: relative blend weights for (nb, logistic, hgb). Default leans on the
    two best calibrated models; nb kept for its hand-reasoned structure."""
    def __init__(self, weights=(1.0, 1.0, 1.0), use_hgb=True):
        self.weights = weights
        self.use_hgb = use_hgb
        self.nb = NaiveBayesTonyModel()
        self.lr = make_pipeline(StandardScaler(),
                                LogisticRegression(penalty="elasticnet", solver="saga",
                                                   l1_ratio=0.5, C=0.2, max_iter=5000))
        self.hgb = HistGradientBoostingClassifier(max_depth=2, learning_rate=0.1,
                                                  max_iter=200) if use_hgb else None
        self._fitted = False

    def fit(self, rows):
        self.nb.fit(rows)
        X, y = _vecy([r for r in rows if r.won_tony is not None])
        self._degenerate = len(set(y)) < 2
        if not self._degenerate:
            self.lr.fit(X, y)
            if self.hgb is not None:
                self.hgb.fit(X, y)
        self._fitted = True
        return self

    def predict_category(self, rows_in_category):
        assert self._fitted, "call fit() first"
        rows_sorted = sorted(rows_in_category, key=lambda r: r.nominee_key)
        # NB probabilities (already a within-race distribution)
        nb_pred = {id(r): p for r, p, _ in self.nb.predict_category(rows_sorted)}
        nb_p = np.array([nb_pred[id(r)] for r in rows_sorted])
        X = _vec(rows_sorted)
        parts = [(self.weights[0], nb_p)]
        if not self._degenerate:
            lr_p = self.lr.predict_proba(X)[:, 1]
            lr_p = lr_p / (lr_p.sum() or 1.0)
            parts.append((self.weights[1], lr_p))
            if self.hgb is not None:
                hg_p = self.hgb.predict_proba(X)[:, 1]
                hg_p = hg_p / (hg_p.sum() or 1.0)
                parts.append((self.weights[2], hg_p))
        wsum = sum(w for w, _ in parts) or 1.0
        blended = sum(w * p for w, p in parts) / wsum
        blended = blended / (blended.sum() or 1.0)
        out = [(r, float(blended[i]), float(blended[i])) for i, r in enumerate(rows_sorted)]
        out.sort(key=lambda t: (-t[1], t[0].nominee_key))
        return out


if __name__ == "__main__":
    from collections import defaultdict
    from validate import load_all_seasons, season_to_rows

    seasons = load_all_seasons()
    data = {s["year"]: season_to_rows(s) for s in seasons
            if any(r.won_tony for r in season_to_rows(s))}
    years = sorted(data)

    def loyo(make):
        hits = tot = 0
        briers = []
        for held in years:
            train = [r for y in years if y != held for r in data[y]]
            mdl = make().fit(train)
            bc = defaultdict(list)
            for r in data[held]:
                bc[r.category].append(r)
            for cat, rs in bc.items():
                if not any(x.won_tony for x in rs):
                    continue
                pred = mdl.predict_category(rs)
                top = pred[0][0]
                hits += 1 if (top.won_tony and top.won_tony >= 0.5) else 0
                tot += 1
                probs = np.array([p for _, p, _ in pred])
                probs = probs / (probs.sum() or 1.0)
                keys = [r for r, _, _ in pred]
                win_i = next(i for i, r in enumerate(keys) if r.won_tony and r.won_tony >= 0.5)
                briers.append(float(np.sum((probs - np.eye(len(keys))[win_i]) ** 2)))
        return hits, tot, np.mean(briers)

    print("LOYO comparison (accuracy | Brier):\n" + "=" * 50)
    h, t, b = loyo(lambda: NaiveBayesTonyModel())
    print(f"  naive-Bayes (current)        {100*h/t:.1f}%  |  {b:.3f}")
    for w, label in [((1, 1, 1), "NB+LR+HGB equal"),
                     ((0.5, 1, 1), "NB(0.5)+LR+HGB"),
                     ((1, 2, 1), "NB+LR(2x)+HGB"),
                     ((0, 1, 1), "LR+HGB (no NB)")]:
        h, t, b = loyo(lambda w=w: EnsembleTonyModel(weights=w))
        print(f"  ensemble {label:20s} {100*h/t:.1f}%  |  {b:.3f}")
