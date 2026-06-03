"""
Hierarchical partial-pooling variant of the model.

The base model (model.py) pools signal weights by a hard category GROUP
(production / acting / creative / design). That's a blunt choice: it assumes,
say, all four design categories share identical signal reliability, and that a
design weight tells you nothing about an acting weight.

This variant does partial pooling instead. For each (category, signal) it
computes a per-category log-likelihood ratio AND the group-level LLR, then
shrinks the per-category estimate toward the group mean by an amount that
depends on how much data the category has:

    llr_final = (n_cat * llr_cat + K * llr_group) / (n_cat + K)

where n_cat is the category's effective sample size and K is a shrinkage
constant. Categories with lots of consistent history keep their own signal;
sparse/noisy categories borrow strength from their group. This is the
statistically-correct middle ground between "one model per category" (overfits)
and "one model per group" (too rigid) - the classic James-Stein / empirical-
Bayes shrinkage idea.

Whether it actually beats the simpler hard-pooling on only ~16-25 seasons is an
empirical question - experiments_v3.py backtests it head to head. We adopt it
only if it wins.
"""
import math
from collections import defaultdict

from model import SIGNALS, CATEGORY_GROUP, ALPHA, LLR_CLAMP


class HierarchicalTonyModel:
    def __init__(self, shrink_k: float = 8.0):
        # K = strength of pull toward the group mean. Higher = more pooling
        # (closer to the base model); lower = trust per-category estimates more.
        self.shrink_k = shrink_k
        self.llr = {}          # llr[category][signal]
        self._fitted = False

    def _group(self, category):
        return CATEGORY_GROUP.get(category, "production")

    def fit(self, rows):
        # tally per-category and per-group signal/win counts in one pass
        cat_counts = defaultdict(lambda: defaultdict(lambda: [0.0, 0.0, 0.0, 0.0]))
        grp_counts = defaultdict(lambda: defaultdict(lambda: [0.0, 0.0, 0.0, 0.0]))
        cat_n = defaultdict(float)
        for r in rows:
            if r.won_tony is None:
                continue
            cat, grp = r.category, self._group(r.category)
            win = r.won_tony >= 0.5
            cat_n[cat] += 1
            for s in SIGNALS:
                v = getattr(r, s, None)
                if v is None:
                    continue
                for store in (cat_counts[cat][s], grp_counts[grp][s]):
                    if win:
                        store[1] += 1
                        if v >= 0.5:
                            store[0] += 1
                    else:
                        store[3] += 1
                        if v >= 0.5:
                            store[2] += 1

        def llr(counts):
            sw, w, sl, l = counts
            p1 = (sw + ALPHA) / (w + 2 * ALPHA)
            p0 = (sl + ALPHA) / (l + 2 * ALPHA)
            return max(-LLR_CLAMP, min(LLR_CLAMP, math.log(p1 / p0)))

        grp_llr = {g: {s: llr(c) for s, c in sigs.items()}
                   for g, sigs in grp_counts.items()}

        self.llr = {}
        for cat, sigs in cat_counts.items():
            grp = self._group(cat)
            n = cat_n[cat]
            self.llr[cat] = {}
            for s in SIGNALS:
                lc = llr(sigs[s]) if s in sigs else 0.0
                lg = grp_llr.get(grp, {}).get(s, 0.0)
                # shrink the per-category estimate toward the group mean
                self.llr[cat][s] = (n * lc + self.shrink_k * lg) / (n + self.shrink_k)
        self._grp_llr = grp_llr
        self._fitted = True
        return self

    def _score_row(self, r):
        weights = self.llr.get(r.category)
        if weights is None:
            # unseen category -> fall back to its group mean
            weights = self._grp_llr.get(self._group(r.category), {})
        score = 0.0
        for s in SIGNALS:
            v = getattr(r, s, None)
            if v is not None and v >= 0.5:
                score += weights.get(s, 0.0)
        return score

    def predict_category(self, rows_in_category):
        assert self._fitted, "call fit() first"
        rows_sorted = sorted(rows_in_category, key=lambda r: r.nominee_key)
        scored = [(r, self._score_row(r)) for r in rows_sorted]
        m = max(s for _, s in scored) if scored else 0.0
        exps = [(r, math.exp(s - m)) for r, s in scored]
        total = sum(e for _, e in exps) or 1.0
        out = [(r, e / total, raw) for (r, e), (_, raw) in zip(exps, scored)]
        out.sort(key=lambda t: (-t[1], t[0].nominee_key))
        return out


if __name__ == "__main__":
    from collections import defaultdict as dd
    from validate import load_all_seasons, season_to_rows
    from model import NaiveBayesTonyModel

    seasons = load_all_seasons()
    rby = {s["year"]: season_to_rows(s) for s in seasons if any(r.won_tony for r in season_to_rows(s))}
    years = sorted(rby)

    def loyo(make):
        hits = tot = 0
        for held in years:
            train = [r for y in years if y != held for r in rby[y]]
            mdl = make().fit(train)
            bc = dd(list)
            for r in rby[held]:
                bc[r.category].append(r)
            for cat, rs in bc.items():
                if not any(x.won_tony for x in rs):
                    continue
                hits += 1 if mdl.predict_category(rs)[0][0].won_tony else 0
                tot += 1
        return hits, tot

    hb, tb = loyo(lambda: NaiveBayesTonyModel())
    print(f"base (hard group-pooling)     {100*hb/tb:.1f}%  ({hb}/{tb})")
    for k in (4.0, 8.0, 16.0):
        hh, th = loyo(lambda: HierarchicalTonyModel(shrink_k=k))
        print(f"hierarchical (K={k:<4})          {100*hh/th:.1f}%  ({hh}/{th})")
