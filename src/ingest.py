"""
Transform raw collected award data into season_<year>.json files in our schema,
applying winner corrections and computing the documentary `winner` key.

Raw input lists Tony data as [{category, nominees}]; our season files use a dict
keyed by category id plus a winner key. This bridges the two and folds in any
independent winner-verification corrections.

Usage:
  ingest_season(season_obj, verification_obj_or_None) -> writes season_<year>.json
We also run structural validation and report any category with no winner or
duplicate winners, so bad data is caught at ingest time, not at model time.
"""
import json
import os

from schema import norm
from categories import TONY_CATEGORIES, PRECURSORS

RAW = os.path.join(os.path.dirname(__file__), "..", "data", "raw")


def _winner_key(nominee):
    if nominee.get("person"):
        return f"{norm(nominee['person'])}|{norm(nominee['show'])}"
    return norm(nominee["show"])


def ingest_season(season, verification=None):
    """season: dict of collected season data (year, tony[], precursors{}).
       verification: optional dict of winner corrections, or None.
       Returns (path_written, list_of_warnings)."""
    year = season["year"]
    warnings = []
    if not season.get("verified_year_matches", True):
        warnings.append(f"{year}: year mismatch flagged -> {season.get('notes','')}")

    # apply winner corrections from the verification pass
    corrections = {}
    if verification and not verification.get("all_correct", True):
        for c in verification.get("corrections", []):
            corrections[c["category"]] = (c["correct_winner_show"],
                                          c.get("correct_winner_person"))

    out = {"year": year,
           "_source_notes": season.get("notes", ""),
           "tony": {cat: {"nominees": [], "winner": None} for cat in TONY_CATEGORIES},
           "precursors": {p: [] for p in PRECURSORS}}

    seen_cats = set()
    for entry in season.get("tony", []):
        cat = entry["category"]
        if cat not in TONY_CATEGORIES:
            warnings.append(f"{year}: unknown category '{cat}' skipped")
            continue
        seen_cats.add(cat)
        nominees = [dict(show=n["show"], person=n.get("person"), won=bool(n.get("won")))
                    for n in entry["nominees"]]
        # apply correction: flip won flags to the corrected winner
        if cat in corrections:
            cshow, cperson = corrections[cat]
            for n in nominees:
                match = norm(n["show"]) == norm(cshow) and (
                    not cperson or norm(n.get("person") or "") == norm(cperson))
                n["won"] = match
            warnings.append(f"{year}/{cat}: winner corrected to {cperson or ''} {cshow}")
        winners = [n for n in nominees if n["won"]]
        if len(winners) == 0 and nominees:
            warnings.append(f"{year}/{cat}: NO winner marked")
        elif len(winners) > 1:
            warnings.append(f"{year}/{cat}: {len(winners)} winners marked (tie? verify)")
        out["tony"][cat]["nominees"] = nominees
        if winners:
            out["tony"][cat]["winner"] = _winner_key(winners[0])

    missing = set(TONY_CATEGORIES) - seen_cats
    if missing:
        warnings.append(f"{year}: categories absent from source: {sorted(missing)}")

    for p in PRECURSORS:
        out["precursors"][p] = season.get("precursors", {}).get(p, [])
    if not out["precursors"]["occ"]:
        warnings.append(f"{year}: OCC data empty")

    path = os.path.join(RAW, f"season_{year}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    return path, warnings


def ingest_batch(results):
    """results: list of [{season, verification}, ...] collected season records."""
    all_warnings = []
    for item in results:
        if not item:
            continue
        _, w = ingest_season(item["season"], item.get("verification"))
        all_warnings.extend(w)
    return all_warnings


if __name__ == "__main__":
    # allow piping a JSON file of collected season records in for batch ingest
    import sys
    if len(sys.argv) > 1:
        with open(sys.argv[1]) as f:
            results = json.load(f)
        for w in ingest_batch(results):
            print("WARN:", w)
    else:
        print("Usage: ingest.py <collected_seasons.json>")
