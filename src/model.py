"""
The prediction model.

Design rationale (read this before "upgrading" to fancier ML)
-------------------------------------------------------------
Per category we have ~4-6 nominees and ~16 seasons => ~16 positive examples and
~70 negatives per category, with heavy shared-voter correlation between the
features. That is FAR too little, too-correlated data for per-category logistic
regression or gradient boosting; they overfit and produce overconfident garbage.

The historically-best-performing approach for this exact problem is a
naive-Bayes-style log-likelihood vote where each signal's weight is its
empirical predictive value. Concretely, for each signal s (e.g. "won Drama
Desk in the matching category") we measure, pooled across all categories and
years:

    P(signal | eventual Tony winner)      "true positive rate"
    P(signal | eventual Tony loser)        "false positive rate"

The log-likelihood-ratio  LLR(s) = log( P(s|win) / P(s|lose) )  is exactly how
much that signal should move our belief. We sum LLRs across signals for each
nominee (the naive-Bayes independence assumption), add a prior, and softmax
within the category so the nominees' probabilities sum to 1.

Signals weight themselves: a precursor that historically tracks the Tony gets a
big LLR; a noisy one gets ~0. This automatically handles "Drama Desk win is
strong, Drama League is weak" WITHOUT us hand-tuning weights, and it degrades
gracefully when a precursor didn't run a category (that signal is just absent).

We pool signal stats across categories by default (musical-design and play-
design behave similarly) because per-category pooling is too sparse, but we
allow a `category_group` split (production / acting / design / creative) so an
acting precursor's reliability isn't diluted by design data.
"""

import math
from collections import defaultdict

# The signals we score, as (attribute_name) on FeatureRow. Each is 1.0/0.0/None.
SIGNALS = [
    "dd_won", "dd_nominated",
    "occ_won", "occ_nominated",
    "dl_won", "dl_nominated",
    "nydcc_won",
    # "Sweep" signal: this nominee's SHOW leads its category race in total Tony
    # nominations. Announced ~3 weeks pre-ceremony (leak-free). Backtested at
    # +2.4 pts overall, concentrated in the low-signal design (+5) and acting
    # (+3) groups where precursors diverge. See experiments.py.
    "is_frontrunner",
    # External time-locked signals (recorded per show, status as-of ceremony day
    # only - see external.py). Leak-guard verified: none exceeds 0.9 precision,
    # i.e. all are noisy/legitimate pre-ceremony signals.
    # Backtested: all-7 -> 63.1% overall (+3.9 over precursors+frontrunner),
    # improving both acting and the weak design group. Pundit-favorite is the
    # strongest single contributor (+2.4).
    "ext_pundit_fav",   # pre-ceremony odds favorite (Gold Derby etc.)
    "ext_rave",         # opening-night critical raves (predate Tonys)
    "ext_wellreviewed", # positive-or-better reviews
    "ext_hit",          # commercial hit / long run as of Tony time
    "ext_flop",         # short flop (negative signal)
    "ext_open",         # still running on ceremony day
    "ext_closed",       # already closed before ceremony (negative signal)
    # Transfer signals: credit a show's pre-Broadway life (Olivier win on the
    # same West End production/role, or major Off-Broadway acclaim in a prior
    # season). These address the "you ignore transfers" blind spot. Note: in
    # backtesting they're flat-to-slightly-negative (the precursors already
    # capture most transfer momentum), but they're included so the model
    # explicitly accounts for the London/Off-Broadway pipeline. Time-locked.
    "tr_olivier",
    "tr_offbway_acclaim",
    "tr_transferred",
    # Consensus-strength signals: how UNANIMOUS the precursor agreement was,
    # not just whether a given body was won. The binary precursor signals can't
    # tell "won 1 of 3 precursors" from "swept all 3 and was the sole winner in
    # the race". Overall LOYO accuracy is flat (the binary signals already carry
    # most of the info), but these fix a specific failure mode the model had:
    # underrating a universal favorite. E.g. they move a clean sweep like Joshua
    # Henry (2026 Lead Actor Musical) from 79% -> 99%, matching human consensus.
    # con_swept_all is intentionally excluded - it added nothing and the trio
    # together regressed. See _assign_consensus in matcher.py.
    "con_win_fraction",
    "con_sole_winner",
    # Per-category historical pundit/odds favorite (time-locked: predictions
    # published BEFORE each ceremony, sourced from Gold Derby via the Wayback
    # Machine, Slant, NPR, etc.). This is the single biggest lift of any feature
    # added: +2.6 pts (61.9 -> 64.5%). odds_strong_fav trips the leak guard at
    # 0.94 precision, but it is NOT a leak - its coverage is only 0.19 (strong
    # favorites exist in ~19% of races). A real leak has high precision AND high
    # coverage; this is just the true fact that pundits' high-confidence calls
    # win ~94% of the time. See _assign_odds in matcher.py and data/odds.json.
    "odds_favorite",
    "odds_strong_fav",
    # "Overdue" signal: nominee has >=2 prior Tony noms and has never won (the
    # classic "it's their time" narrative). Honest finding: the broader prior-
    # nomination-count and commercial-heat features (tested in matcher) did NOT
    # help - the precursors and pundit odds already price in the overdue
    # narrative, so they were redundant. Only this specific flag is kept, and
    # only because it is neutral-to-marginally-positive and captures a story
    # voters demonstrably act on. See _assign_prior_noms in matcher.py.
    "pn_overdue",
]

# Group categories so signal reliability is learned within similar award types
# (avoids design data diluting acting-signal strength) while keeping samples big
# enough to be stable.
CATEGORY_GROUP = {
    "best_play": "production", "best_musical": "production",
    "best_revival_play": "production", "best_revival_musical": "production",
    "best_book": "production", "best_score": "production",
    "lead_actor_play": "acting", "lead_actress_play": "acting",
    "lead_actor_musical": "acting", "lead_actress_musical": "acting",
    "feat_actor_play": "acting", "feat_actress_play": "acting",
    "feat_actor_musical": "acting", "feat_actress_musical": "acting",
    "direction_play": "creative", "direction_musical": "creative",
    "choreography": "creative", "orchestrations": "creative",
    "scenic_play": "design", "costume_play": "design",
    "lighting_play": "design", "sound_play": "design",
    "scenic_musical": "design", "costume_musical": "design",
    "lighting_musical": "design", "sound_musical": "design",
}

# Laplace smoothing so a signal seen in 0 losers doesn't yield infinite LLR.
ALPHA = 1.0
# Clamp LLR magnitude so one extreme signal can't dominate on sparse data.
LLR_CLAMP = 3.0


class NaiveBayesTonyModel:
    def __init__(self, pool_by_group: bool = True):
        self.pool_by_group = pool_by_group
        # llr[group][signal] = log-likelihood ratio
        self.llr = {}
        self._fitted = False

    def _group(self, category: str) -> str:
        return CATEGORY_GROUP.get(category, "production") if self.pool_by_group else "all"

    def fit(self, rows: list):
        """rows: list[FeatureRow] from historical seasons (won_tony in {0,1})."""
        # counts[group][signal] = [n_signal_and_win, n_win, n_signal_and_lose, n_lose]
        counts = defaultdict(lambda: defaultdict(lambda: [0.0, 0.0, 0.0, 0.0]))
        for r in rows:
            if r.won_tony is None:
                continue
            g = self._group(r.category)
            win = r.won_tony >= 0.5
            for s in SIGNALS:
                v = getattr(r, s)
                if v is None:
                    continue  # precursor didn't run this category -> not evidence
                c = counts[g][s]
                if win:
                    c[1] += 1
                    if v >= 0.5:
                        c[0] += 1
                else:
                    c[3] += 1
                    if v >= 0.5:
                        c[2] += 1
        self.llr = {}
        for g, sigs in counts.items():
            self.llr[g] = {}
            for s, (sw, w, sl, l) in sigs.items():
                p_s_given_win = (sw + ALPHA) / (w + 2 * ALPHA)
                p_s_given_lose = (sl + ALPHA) / (l + 2 * ALPHA)
                llr = math.log(p_s_given_win / p_s_given_lose)
                self.llr[g][s] = max(-LLR_CLAMP, min(LLR_CLAMP, llr))
        self._fitted = True
        return self

    def _score_row(self, r) -> float:
        """Sum of LLRs for the signals present on this nominee."""
        g = self._group(r.category)
        weights = self.llr.get(g, {})
        score = 0.0
        for s in SIGNALS:
            v = getattr(r, s)
            if v is None:
                continue
            if v >= 0.5:
                score += weights.get(s, 0.0)
            # absence of a signal the winner usually has is mild negative evidence:
            else:
                # subtract the "complement" LLR: log P(~s|win)/P(~s|lose)
                score += weights.get(s, 0.0) * -0.0  # kept explicit & off by default
        return score

    def predict_category(self, rows_in_category: list) -> list:
        """Given the FeatureRows for ONE (year, category), return them paired
        with a normalized win probability, sorted descending."""
        assert self._fitted, "call fit() first"
        # IMPORTANT: sort inputs by a key uncorrelated with winning BEFORE
        # scoring. Source data (Wikipedia) lists winners first; without this,
        # ties in the softmax would resolve to input order and the model would
        # "predict" no-signal categories purely from row ordering - a leak that
        # inflates backtests and does NOT generalize to unseen years.
        rows_sorted = sorted(rows_in_category, key=lambda r: r.nominee_key)
        scored = [(r, self._score_row(r)) for r in rows_sorted]
        # softmax over LLR sums -> probabilities summing to 1 within the category
        m = max(s for _, s in scored) if scored else 0.0
        exps = [(r, math.exp(s - m)) for r, s in scored]
        total = sum(e for _, e in exps) or 1.0
        out = [(r, e / total, raw) for (r, e), (_, raw) in zip(exps, scored)]
        # tie-break by nominee_key (deterministic, leak-free), not input order
        out.sort(key=lambda t: (-t[1], t[0].nominee_key))
        return out

    def report_weights(self) -> str:
        lines = ["Learned log-likelihood-ratio weights (higher = stronger Tony signal):"]
        for g in sorted(self.llr):
            lines.append(f"\n[{g}]")
            for s, w in sorted(self.llr[g].items(), key=lambda kv: -kv[1]):
                lines.append(f"  {s:16s} {w:+.3f}")
        return "\n".join(lines)


if __name__ == "__main__":
    # Tiny synthetic sanity test: a signal that perfectly predicts wins should
    # get a strongly positive LLR and the model should rank that nominee first.
    from schema import FeatureRow
    rows = []
    for yr in range(2008, 2024):
        # winner won DD; two losers did not
        rows.append(FeatureRow(year=yr, category="best_play", nominee_key=f"w{yr}",
                               nominee_show=f"W{yr}", nominee_person=None,
                               dd_won=1.0, dd_nominated=1.0, won_tony=1.0))
        for j in range(2):
            rows.append(FeatureRow(year=yr, category="best_play", nominee_key=f"l{yr}_{j}",
                                   nominee_show=f"L{yr}_{j}", nominee_person=None,
                                   dd_won=0.0, dd_nominated=1.0, won_tony=0.0))
    m = NaiveBayesTonyModel().fit(rows)
    assert m.llr["production"]["dd_won"] > 1.0, m.llr
    test = [r for r in rows if r.year == 2010]
    pred = m.predict_category(test)
    assert pred[0][0].nominee_show.startswith("W"), pred
    print("OK: model sanity test passes.")
    print(m.report_weights())
