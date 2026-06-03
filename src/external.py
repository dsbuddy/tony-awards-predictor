"""
Integrate time-locked external per-show data (commercial / critical / pundit)
into the feature rows, and TEST whether it adds predictive lift over the
current 59.2% naive-Bayes model.

Includes a LEAK GUARD: any single feature whose presence correlates with
winning above a threshold (~0.9 precision AND high coverage) is flagged as a
probable time-leak, because genuine pre-ceremony signals are noisy. Time-locking
every feature to pre-ceremony status is essential to avoid biasing the backtest.
"""
import json
import os
from collections import defaultdict

from schema import norm
from validate import load_all_seasons
from experiments_v2 import enrich, run_naive_bayes
import model as M

EXT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "external_shows.json")


def load_external():
    """Return {year: {norm_show: {field: value}}}."""
    if not os.path.exists(EXT_PATH):
        return {}
    with open(EXT_PATH) as f:
        data = json.load(f)
    out = {}
    for season in data:
        ymap = {}
        for s in season.get("shows", []):
            ymap[norm(s["show"])] = s
        out[season["year"]] = ymap
    return out


def attach_external(rby, ext):
    """Add external features to each row, defaulting sensibly when a show is
    missing from the external data."""
    for year, rows in rby.items():
        ymap = ext.get(year, {})
        for r in rows:
            info = ymap.get(norm(r.nominee_show), {})
            # binary/encoded features, all leak-checked downstream
            r.ext_open = _b(info.get("still_open_at_tonys"), default=1.0)
            r.ext_closed = _b(info.get("closed_before_tonys"), default=0.0)
            r.ext_pundit_fav = _b(info.get("is_pundit_favorite"), default=0.0)
            rec = info.get("critics_reception", "unknown")
            r.ext_rave = 1.0 if rec == "rave" else 0.0
            r.ext_wellreviewed = 1.0 if rec in ("rave", "positive") else 0.0
            run = info.get("run_longevity", "unknown")
            r.ext_hit = 1.0 if run == "hit_long_run" else 0.0
            r.ext_flop = 1.0 if run == "short_flop" else 0.0
    return rby


def _b(v, default):
    if v is True:
        return 1.0
    if v is False:
        return 0.0
    return default


def leak_guard(rby, feats):
    """For each feature, among LABELED rows, compute:
       precision = P(win | feature==1), coverage = how many winners have it.
       Flag features that look like leaks (near-perfect precision AND cover most
       winners). Returns a list of warnings."""
    warnings = []
    for f in feats:
        win_with = total_with = winners = 0
        for rows in rby.values():
            for r in rows:
                if r.won_tony is None:
                    continue
                if r.won_tony >= 0.5:
                    winners += 1
                v = getattr(r, f, 0.0)
                if v and v >= 0.5:
                    total_with += 1
                    if r.won_tony >= 0.5:
                        win_with += 1
        if total_with == 0:
            warnings.append(f"  {f}: never present (no signal)")
            continue
        precision = win_with / total_with
        coverage = win_with / winners if winners else 0
        flag = ""
        if precision > 0.9 and coverage > 0.5:
            flag = "  <<< PROBABLE LEAK (too predictive for a pre-ceremony signal)"
        warnings.append(f"  {f:18s} precision={precision:.2f} coverage={coverage:.2f} "
                        f"(n={total_with}){flag}")
    return warnings


EXT_FEATS = ["ext_open", "ext_closed", "ext_pundit_fav", "ext_rave",
             "ext_wellreviewed", "ext_hit", "ext_flop"]


def nb_with(rby, extra):
    orig = M.SIGNALS[:]
    M.SIGNALS[:] = orig + extra
    try:
        h, t = run_naive_bayes(rby)
    finally:
        M.SIGNALS[:] = orig
    return h, t


if __name__ == "__main__":
    seasons = load_all_seasons()
    rby = enrich(seasons)
    ext = load_external()
    if not ext:
        print(f"No external data at {EXT_PATH} yet.")
        raise SystemExit(0)
    attach_external(rby, ext)

    print("LEAK GUARD - are any external features suspiciously predictive?")
    print("=" * 66)
    for w in leak_guard(rby, EXT_FEATS):
        print(w)

    print("\nLIFT TEST (naive-Bayes, frontrunner already baked in) - 59.2% base")
    print("=" * 66)
    h, t = run_naive_bayes(rby)
    print(f"  current model                       {100*h/t:4.1f}%")
    for f in EXT_FEATS:
        h, t = nb_with(rby, [f])
        print(f"  + {f:18s}                {100*h/t:4.1f}%")
    # best combos
    for combo in (["ext_pundit_fav"], ["ext_closed", "ext_wellreviewed"],
                  ["ext_pundit_fav", "ext_flop"], EXT_FEATS):
        h, t = nb_with(rby, combo)
        print(f"  + {', '.join(combo):34s} {100*h/t:4.1f}%")
