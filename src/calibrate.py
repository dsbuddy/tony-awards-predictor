"""
Probability calibration.

The base model's softmax probabilities are well-ordered (higher = more likely to
win) but not perfectly calibrated: backtesting showed it runs over-confident in
the middle of the range (a "60%" pick won closer to 45% of the time). This maps
the raw probabilities onto calibrated ones so a stated 60% actually means ~60%.

Method: isotonic regression (monotonic, non-parametric) fit on OUT-OF-FOLD
predictions only. We never fit the calibrator on probabilities the model
produced for its own training data - we use the leave-one-year-out predictions,
where each year was scored by a model that didn't see it. Fitting on in-sample
probabilities would itself be a leak and would learn an over-optimistic curve.

Because we re-normalize within each category after calibrating (a category's
win probabilities must still sum to 1), this is a *ranking-preserving* rescale:
it changes the numbers, not which nominee is picked. So it improves the honesty
of the confidence levels without changing accuracy - exactly what we want.
"""
import numpy as np
from sklearn.isotonic import IsotonicRegression

from analytics import races, all_nominee_probs


def _all_loyo_nominee_probs():
    return all_nominee_probs()


class ProbabilityCalibrator:
    def __init__(self):
        self.iso = None

    def fit_from_loyo(self):
        """Fit on EVERY nominee's out-of-fold probability vs whether it won,
        not just the top pick. Fitting on top-picks-only and then renormalizing
        within a category badly distorts toss-up races (every nominee's raw prob
        maps high, renormalization then forces the leader to ~100%). Using all
        nominees gives the isotonic curve the full 0..1 support it needs."""
        rows = _all_loyo_nominee_probs()
        x = np.array([p for p, _ in rows])
        y = np.array([w for _, w in rows])
        self.iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        self.iso.fit(x, y)
        return self

    def transform_prob(self, p: float) -> float:
        if self.iso is None:
            return p
        return float(self.iso.predict([p])[0])

    def calibrate_category(self, ranked):
        """ranked: list of (row, prob, raw) from model.predict_category.
        Returns the same list with calibrated probabilities that still sum to 1
        within the category (ranking unchanged)."""
        if self.iso is None or not ranked:
            return ranked
        cal = [max(1e-9, self.transform_prob(p)) for _, p, _ in ranked]
        total = sum(cal) or 1.0
        out = [(r, c / total, raw) for (r, _, raw), c in zip(ranked, cal)]
        out.sort(key=lambda t: (-t[1], t[0].nominee_key))
        return out


# module-level singleton so callers don't refit repeatedly
_CAL = None


def get_calibrator():
    global _CAL
    if _CAL is None:
        _CAL = ProbabilityCalibrator().fit_from_loyo()
    return _CAL


if __name__ == "__main__":
    from collections import defaultdict
    cal = get_calibrator()
    # show the mapping at a few points
    print("Raw -> calibrated probability mapping:")
    for p in (0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 0.99):
        print(f"  {p*100:4.0f}%  ->  {cal.transform_prob(p)*100:4.0f}%")

    # verify it improves calibration error on the LOYO races
    rs = races()
    def ece(get_p):
        # expected calibration error over 10 bins
        bins = defaultdict(lambda: [0.0, 0.0, 0])
        for r in rs:
            p = get_p(r["prob"])
            b = min(int(p * 10), 9)
            bins[b][0] += p
            bins[b][1] += 1.0 if r["correct"] else 0.0
            bins[b][2] += 1
        n = len(rs)
        err = 0.0
        for b, (sp, sc, c) in bins.items():
            if c:
                err += (c / n) * abs(sp / c - sc / c)
        return err
    raw_ece = ece(lambda p: p)
    cal_ece = ece(lambda p: cal.transform_prob(p))
    print(f"\nExpected calibration error (lower = better):")
    print(f"  raw:        {raw_ece*100:.1f}%")
    print(f"  calibrated: {cal_ece*100:.1f}%")
