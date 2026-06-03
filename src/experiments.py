"""
Experiment harness: test candidate model improvements against the honest
leave-one-year-out baseline (57%). Each experiment is evaluated the SAME way
as validate.py so deltas are comparable and leak-free.

Candidates tested:
  A. Frontrunner feature  - flag the nominee whose SHOW has the most total Tony
     nominations in that season (the "sweep" signal).
  B. Precursor unanimity  - flag nominees that won 2+ precursors (consensus).
  C. Recency weighting    - weight recent training seasons more heavily.

We compute everything from existing season files; no new data needed.
"""
from collections import defaultdict
from validate import load_all_seasons, season_to_rows
from model import NaiveBayesTonyModel, SIGNALS, CATEGORY_GROUP
import math


def show_nom_counts(season):
    """How many Tony categories each show is nominated in, this season."""
    counts = defaultdict(int)
    for cat, data in season["tony"].items():
        seen = set()
        for n in data["nominees"]:
            from schema import norm
            s = norm(n["show"])
            if s not in seen:        # don't double-count a show w/ 2 noms in one cat
                counts[s] += 1
                seen.add(s)
    return counts


def annotate_frontrunner(seasons):
    """Attach .show_nom_count and .is_frontrunner (max in its category race) to
    each FeatureRow, returning {year: [rows]}."""
    from schema import norm
    out = {}
    for s in seasons:
        if not any(r.won_tony for r in season_to_rows(s)):
            continue
        counts = show_nom_counts(s)
        rows = season_to_rows(s)
        bycat = defaultdict(list)
        for r in rows:
            r.show_nom_count = counts.get(norm(r.nominee_show), 0)
            bycat[r.category].append(r)
        for cat, rs in bycat.items():
            mx = max((r.show_nom_count for r in rs), default=0)
            for r in rs:
                # frontrunner only meaningful when there's separation
                r.is_frontrunner = 1.0 if (r.show_nom_count == mx and mx > 0
                                           and sum(x.show_nom_count == mx for x in rs) < len(rs)) else 0.0
        out[s["year"]] = rows
    return out


class ExtendedModel(NaiveBayesTonyModel):
    """Adds optional extra binary signals + optional recency weighting."""
    def __init__(self, extra_signals=(), recency_halflife=None, **kw):
        super().__init__(**kw)
        self.extra = list(extra_signals)
        self.recency_halflife = recency_halflife
        self._max_year = None

    def fit(self, rows):
        self._all_signals = SIGNALS + self.extra
        self._max_year = max((r.year for r in rows), default=0)
        counts = defaultdict(lambda: defaultdict(lambda: [0.0, 0.0, 0.0, 0.0]))
        for r in rows:
            if r.won_tony is None:
                continue
            w_recent = 1.0
            if self.recency_halflife:
                age = self._max_year - r.year
                w_recent = 0.5 ** (age / self.recency_halflife)
            g = self._group(r.category)
            win = r.won_tony >= 0.5
            for s in self._all_signals:
                v = getattr(r, s, None)
                if v is None:
                    continue
                c = counts[g][s]
                if win:
                    c[1] += w_recent
                    if v >= 0.5: c[0] += w_recent
                else:
                    c[3] += w_recent
                    if v >= 0.5: c[2] += w_recent
        self.llr = {}
        for g, sigs in counts.items():
            self.llr[g] = {}
            for s, (sw, w, sl, l) in sigs.items():
                p1 = (sw + 1.0) / (w + 2.0)
                p0 = (sl + 1.0) / (l + 2.0)
                self.llr[g][s] = max(-3.0, min(3.0, math.log(p1 / p0)))
        self._fitted = True
        return self

    def _score_row(self, r):
        g = self._group(r.category)
        weights = self.llr.get(g, {})
        score = 0.0
        for s in getattr(self, "_all_signals", SIGNALS):
            v = getattr(r, s, None)
            if v is not None and v >= 0.5:
                score += weights.get(s, 0.0)
        return score


def backtest_variant(rows_by_year, model_factory):
    years = sorted(rows_by_year)
    hits = total = 0
    for held in years:
        train = [r for y in years if y != held for r in rows_by_year[y]]
        m = model_factory().fit(train)
        bycat = defaultdict(list)
        for r in rows_by_year[held]:
            bycat[r.category].append(r)
        for cat, rs in bycat.items():
            if not any(x.won_tony for x in rs):
                continue
            pred = m.predict_category(rs)
            hits += 1 if pred[0][0].won_tony else 0
            total += 1
    return hits, total


if __name__ == "__main__":
    seasons = load_all_seasons()
    rows_by_year = annotate_frontrunner(seasons)

    print("LEAVE-ONE-YEAR-OUT BACKTEST - candidate improvements\n" + "=" * 56)

    variants = {
        "BASELINE (current model)":
            lambda: NaiveBayesTonyModel(),
        "A. + Frontrunner (Tony-nom-count)":
            lambda: ExtendedModel(extra_signals=["is_frontrunner"]),
        "B. + Precursor unanimity feature":
            lambda: ExtendedModel(extra_signals=["unanimous"]),
        "A+B. Frontrunner + unanimity":
            lambda: ExtendedModel(extra_signals=["is_frontrunner", "unanimous"]),
        "C. Recency weighting (halflife=6y)":
            lambda: ExtendedModel(recency_halflife=6),
        "A+C. Frontrunner + recency":
            lambda: ExtendedModel(extra_signals=["is_frontrunner"], recency_halflife=6),
    }

    # add the 'unanimous' signal (won 2+ precursors) to rows
    for rows in rows_by_year.values():
        for r in rows:
            r.unanimous = 1.0 if r.n_precursor_wins >= 2 else 0.0

    base_acc = None
    for name, fac in variants.items():
        h, t = backtest_variant(rows_by_year, fac)
        acc = 100 * h / t
        if base_acc is None:
            base_acc = acc
        delta = acc - base_acc
        flag = "  <-- baseline" if delta == 0 else f"  ({delta:+.1f} pts)"
        print(f"  {name:38s} {h}/{t} = {acc:4.1f}%{flag}")
