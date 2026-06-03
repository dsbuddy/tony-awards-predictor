"""
Integrate transfer signals (Olivier wins for West End transfers, prior-season
Off-Broadway acclaim for Off-Broadway transfers) and test whether they add
predictive lift over the current model.

These features address a known blind spot: the base model only looks at
same-season precursors, so a show that built momentum in London or Off-Broadway
before transferring to Broadway gets no credit for it. Like all features here,
every datum is time-locked to before each Tony ceremony, and we run the same
leak guard (a feature that predicts winners too cleanly is probably a leak).
"""
import json
import os

from schema import norm
from validate import load_all_seasons
from experiments_v2 import enrich, run_naive_bayes
from external import load_external, attach_external
import model as M

TR_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "transfers.json")


def load_transfers():
    """Return {year: {norm_show: record}}."""
    if not os.path.exists(TR_PATH):
        return {}
    with open(TR_PATH) as f:
        data = json.load(f)
    out = {}
    for season in data:
        out[season["year"]] = {norm(s["show"]): s for s in season.get("shows", [])}
    return out


def attach_transfers(rby, tr):
    """Add transfer features to each row. Neutral defaults when a show didn't
    transfer (the common case)."""
    for year, rows in rby.items():
        ymap = tr.get(year, {})
        for r in rows:
            info = ymap.get(norm(r.nominee_show), {})
            r.tr_olivier = 1.0 if info.get("won_olivier") is True else 0.0
            r.tr_offbway_acclaim = 1.0 if info.get("prior_offbroadway_acclaim") is True else 0.0
            # generic "transferred at all" flag (transfers tend to be stronger entries)
            tf = info.get("transferred_from", "none")
            r.tr_transferred = 1.0 if tf in ("london", "offbroadway", "both") else 0.0
    return rby


TR_FEATS = ["tr_olivier", "tr_offbway_acclaim", "tr_transferred"]


def leak_guard(rby, feats):
    out = []
    for f in feats:
        win_with = total_with = winners = 0
        for rows in rby.values():
            for r in rows:
                if r.won_tony is None:
                    continue
                if r.won_tony >= 0.5:
                    winners += 1
                if getattr(r, f, 0.0) >= 0.5:
                    total_with += 1
                    if r.won_tony >= 0.5:
                        win_with += 1
        if total_with == 0:
            out.append(f"  {f:20s} never present (no signal in data)")
            continue
        prec = win_with / total_with
        cov = win_with / winners if winners else 0
        flag = "  <<< PROBABLE LEAK" if (prec > 0.9 and cov > 0.5) else ""
        out.append(f"  {f:20s} precision={prec:.2f} coverage={cov:.2f} (n={total_with}){flag}")
    return out


def nb_with(rby, extra):
    orig = M.SIGNALS[:]
    M.SIGNALS[:] = orig + extra
    try:
        return run_naive_bayes(rby)
    finally:
        M.SIGNALS[:] = orig


if __name__ == "__main__":
    seasons = load_all_seasons()
    rby = enrich(seasons)
    attach_external(rby, load_external())   # keep the current external features on
    tr = load_transfers()
    if not tr:
        print(f"No transfer data at {TR_PATH} yet.")
        raise SystemExit(0)
    attach_transfers(rby, tr)

    n_shows = sum(len(v) for v in tr.values())
    print(f"Loaded transfer records for {n_shows} show-seasons across {len(tr)} seasons.\n")

    print("LEAK GUARD")
    print("=" * 60)
    for line in leak_guard(rby, TR_FEATS):
        print(line)

    print("\nLIFT TEST (on top of the full current model)")
    print("=" * 60)
    h, t = run_naive_bayes(rby)
    print(f"  current model (no transfer feats)   {h}/{t} = {100*h/t:.1f}%")
    for f in TR_FEATS:
        h, t = nb_with(rby, [f])
        print(f"  + {f:20s}             {h}/{t} = {100*h/t:.1f}%")
    h, t = nb_with(rby, TR_FEATS)
    print(f"  + all transfer feats                {h}/{t} = {100*h/t:.1f}%")
    # focus: do they help the PLAY categories (where London transfers cluster)?
    print("\n  (Olivier transfers cluster in play categories; Off-Bway in musicals)")
