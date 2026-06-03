"""
Entity matching + feature extraction.

Given a season's Tony nominees and the precursor results, produce one
FeatureRow per (category, nominee) recording which precursors nominated/won
that exact entity.

The two hard problems this solves:

1. Category alignment - use CROSSWALK to find which precursor categories
   correspond to a given Tony category, including combined Play+Musical design
   awards (routed by show type) and the Score->Music+Lyrics split.

2. Entity alignment - decide whether a precursor result refers to the same
   show/person as a Tony nominee, using normalized fuzzy matching, AND only
   counting it if the precursor result is Broadway-eligible (so an Off-Broadway
   Drama Desk winner doesn't get scored against a Tony category).
"""

from schema import norm, FeatureRow
from categories import (CROSSWALK, is_musical_category, OFFBROADWAY_RISK_PRECURSORS,
                        match_mode)


def _crosswalk_entry(cat_id: str, precursor: str):
    """Return (list_of_precursor_category_names, combined_flag) for this
    Tony category + precursor."""
    raw = CROSSWALK[cat_id][precursor]
    if isinstance(raw, dict):
        return [norm(n) for n in raw["names"]], raw.get("combined", False)
    return [norm(n) for n in raw], False


def _show_of(key: str) -> str:
    return key.split("|", 1)[1] if "|" in key else key


def _person_of(key: str) -> str:
    return key.split("|", 1)[0] if "|" in key else ""


def _show_match(a: str, b: str) -> bool:
    """Normalized show equality, tolerant of subtitle/article differences."""
    return a == b or (len(a) > 3 and len(b) > 3 and (a in b or b in a))


def _entity_matches(nominee_key: str, result_key: str, mode: str) -> bool:
    """Decide if a precursor result refers to the same entity as a Tony nominee.

    mode == "person": match on the PERSON (acting categories - one show has
        several nominees, so the show alone can't disambiguate). We still guard
        against a clearly conflicting show when both list one.
    mode == "show": match on the PRODUCTION (design/creative/production awards).
        Design TEAMS are credited differently across award bodies (e.g. Tony's
        'Miriam Buether and 59 Productions' vs Drama Desk's 'Miriam Buether,
        Jamie Harrison, and Chris Fisher'), so person-matching is wrong here - the show is the stable key."""
    if mode == "person":
        n_person, r_person = _person_of(nominee_key), _person_of(result_key)
        if not n_person or not r_person:
            return False
        if n_person != r_person:
            # tolerate ordering/extra-name differences ("Sutton Foster" subset)
            return n_person in r_person or r_person in n_person
        n_show, r_show = _show_of(nominee_key), _show_of(result_key)
        if n_show and r_show and not _show_match(n_show, r_show):
            return False
        return True
    # mode == "show"
    return _show_match(_show_of(nominee_key), _show_of(result_key))


def _is_eligible(result, precursor: str) -> bool:
    """Should this precursor result count toward a Tony category? For
    Off-Broadway-risk precursors we require is_broadway is not False. If the
    flag is unknown (None) we allow it but the data-quality report flags it."""
    if precursor in OFFBROADWAY_RISK_PRECURSORS and result.get("is_broadway") is False:
        return False
    return True


def build_features(season: dict, precursor_objs: dict) -> list:
    """season: the dict for one year (see schema.make_season / loaded JSON).
       precursor_objs: {precursor_name: [result_dict, ...]} from season['precursors'].
       Returns list[FeatureRow]."""
    rows = []
    for cat_id, catdata in season["tony"].items():
        musical = is_musical_category(cat_id)
        mode = match_mode(cat_id)
        for nominee in catdata["nominees"]:
            nom_key = _nominee_key(nominee)
            fr = FeatureRow(
                year=season["year"],
                category=cat_id,
                nominee_key=nom_key,
                nominee_show=nominee["show"],
                nominee_person=nominee.get("person"),
                won_tony=1.0 if nominee.get("won") else 0.0,
            )
            for precursor in ("drama_desk", "occ", "drama_league", "nydcc"):
                names, combined = _crosswalk_entry(cat_id, precursor)
                if not names:
                    continue  # precursor has no equivalent category
                nominated = won = False
                for result in precursor_objs.get(precursor, []):
                    rcat = norm(result["category"])
                    if rcat not in names:
                        continue
                    # Combined-award routing: only needed when the precursor
                    # category name does NOT itself encode play/musical. If the
                    # result carries a show_type, use it to route; if it doesn't
                    # AND the matched name is a generic combined one, fall back
                    # to show-type inference is impossible -> rely on show match.
                    if combined and result.get("show_type") is not None:
                        if (result["show_type"] == "musical") != musical:
                            continue  # combined award routed to wrong show-type
                    if not _is_eligible(result, precursor):
                        continue
                    if _entity_matches(nom_key, _result_key(result), mode):
                        nominated = True
                        if result.get("won"):
                            won = True
                _assign_signal(fr, precursor, nominated, won)
            _finalize(fr)
            rows.append(fr)
    _assign_frontrunner(season, rows)
    _assign_external(season, rows)
    _assign_transfers(season, rows)
    _assign_consensus(rows)
    _assign_odds(season, rows)
    _assign_prior_noms(season, rows)
    _assign_grosses(season, rows)
    return rows


_PN_CACHE = None
_HEAT_CACHE = None


def _assign_prior_noms(season, rows):
    """Attach 'overdue'/prior-career signals for acting + directing nominees:
    how many prior Tony noms a person had, whether they've never won despite
    multiple noms (the classic 'it's their time' narrative), and whether they're
    a past winner. Fixed historical facts (time-lock trivially safe). See
    data/prior_noms.json."""
    global _PN_CACHE
    if _PN_CACHE is None:
        import json, os
        path = os.path.join(os.path.dirname(__file__), "..", "data", "prior_noms.json")
        _PN_CACHE = {}
        if os.path.exists(path):
            with open(path) as f:
                for s in json.load(f):
                    by_person = {}
                    for n in s.get("nominees", []):
                        by_person[(n["category"], norm(n["person"]))] = n
                    _PN_CACHE[s["year"]] = by_person
    pmap = _PN_CACHE.get(season["year"], {})
    if not pmap:
        return
    for r in rows:
        if not r.nominee_person:
            continue
        info = pmap.get((r.category, norm(r.nominee_person)))
        if not info:
            # try person-only match within same category family
            for (cat, person), n in pmap.items():
                if cat == r.category and (person in norm(r.nominee_person)
                                          or norm(r.nominee_person) in person):
                    info = n
                    break
        if info:
            # normalized prior noms: cap at 5 so a veteran doesn't dominate
            r.pn_prior_noms = min(info.get("prior_noms", 0), 5) / 5.0
            r.pn_overdue = 1.0 if info.get("never_won_but_nommed") else 0.0
            r.pn_prior_winner = 1.0 if info.get("prior_wins", 0) > 0 else 0.0


def _assign_grosses(season, rows):
    """Attach pre-ceremony commercial heat (per show) and the Astaire/Chita
    Rivera choreography award winner (an extra precursor the usual four miss).
    Time-locked to before the ceremony. See data/grosses_astaire.json."""
    global _HEAT_CACHE
    if _HEAT_CACHE is None:
        import json, os
        path = os.path.join(os.path.dirname(__file__), "..", "data", "grosses_astaire.json")
        _HEAT_CACHE = {}
        if os.path.exists(path):
            with open(path) as f:
                for s in json.load(f):
                    _HEAT_CACHE[s["year"]] = {
                        "heat": {norm(x["show"]): x for x in s.get("shows", [])},
                        "astaire": s.get("astaire_choreo_winner"),
                    }
    entry = _HEAT_CACHE.get(season["year"])
    if not entry:
        return
    heat = entry["heat"]
    astaire = entry["astaire"]
    astaire_show = norm(astaire["show"]) if astaire else None
    for r in rows:
        info = heat.get(norm(r.nominee_show))
        if info:
            h = info.get("commercial_heat")
            r.heat_hot = 1.0 if h == "hot" else 0.0
            r.heat_soft = 1.0 if h == "soft" else 0.0
        # Astaire choreography award only informs the choreography category
        if astaire_show and r.category == "choreography" and _show_match(astaire_show, norm(r.nominee_show)):
            r.astaire_winner = 1.0


# odds data loaded once, cached
_ODDS_CACHE = None

# map the odds-data category labels onto our internal category ids
_ODDS_LABEL_TO_CAT = {
    "best play": "best_play", "best musical": "best_musical",
    "best revival of a play": "best_revival_play",
    "best revival of a musical": "best_revival_musical",
    "best book of a musical": "best_book", "best book": "best_book",
    "best original score": "best_score", "best score": "best_score",
    "best leading actor in a play": "lead_actor_play", "lead actor in a play": "lead_actor_play",
    "best leading actress in a play": "lead_actress_play", "lead actress in a play": "lead_actress_play",
    "best leading actor in a musical": "lead_actor_musical", "lead actor in a musical": "lead_actor_musical",
    "best leading actress in a musical": "lead_actress_musical", "lead actress in a musical": "lead_actress_musical",
    "best featured actor in a play": "feat_actor_play", "featured actor in a play": "feat_actor_play",
    "best featured actress in a play": "feat_actress_play", "featured actress in a play": "feat_actress_play",
    "best featured actor in a musical": "feat_actor_musical", "featured actor in a musical": "feat_actor_musical",
    "best featured actress in a musical": "feat_actress_musical", "featured actress in a musical": "feat_actress_musical",
    "best direction of a play": "direction_play", "direction of a play": "direction_play",
    "best direction of a musical": "direction_musical", "direction of a musical": "direction_musical",
    "best choreography": "choreography", "choreography": "choreography",
    "best orchestrations": "orchestrations", "orchestrations": "orchestrations",
}


def _assign_odds(season, rows):
    """Attach the per-category historical pundit/odds favorite signal. For each
    category that the odds data covers, find which nominee the pre-ceremony
    pundits favored and flag it (and whether it was a STRONG call). Time-locked:
    these are predictions published before each ceremony. Match on person for
    acting/creative, show for production awards. See data/odds.json."""
    global _ODDS_CACHE
    if _ODDS_CACHE is None:
        import json, os
        path = os.path.join(os.path.dirname(__file__), "..", "data", "odds.json")
        _ODDS_CACHE = {}
        if os.path.exists(path):
            with open(path) as f:
                for s in json.load(f):
                    favs = {}
                    for fav in s.get("favorites", []):
                        cat = _ODDS_LABEL_TO_CAT.get(norm(fav["category_label"]))
                        if cat:
                            favs[cat] = fav
                    _ODDS_CACHE[s["year"]] = favs
    favs = _ODDS_CACHE.get(season["year"], {})
    if not favs:
        return
    by_cat = {}
    for r in rows:
        by_cat.setdefault(r.category, []).append(r)
    for cat, rs in by_cat.items():
        fav = favs.get(cat)
        if not fav:
            continue
        fav_show, fav_person = norm(fav.get("show", "")), norm(fav.get("person") or "")
        strong = 1.0 if fav.get("confidence") == "strong" else 0.0
        for r in rs:
            nom_person = norm(r.nominee_person or "")
            if fav_person and nom_person:
                hit = fav_person in nom_person or nom_person in fav_person
            else:
                hit = _show_match(fav_show, norm(r.nominee_show))
            if hit:
                r.odds_favorite = 1.0
                r.odds_strong_fav = strong


def _assign_consensus(rows):
    """Consensus-strength features. The base precursor signals are binary
    (won Drama Desk yes/no), which can't distinguish 'won 1 of 3 precursors'
    from 'swept all 3'. These graded features capture HOW unanimous the
    agreement was - addressing cases where a nominee is the clear universal
    favorite but the binary signals undersell it. Computed per (category) race."""
    by_cat = {}
    for r in rows:
        by_cat.setdefault(r.category, []).append(r)
    for cat, rs in by_cat.items():
        # how many distinct precursors ran this category at all (any nominee
        # won or was nominated) - the denominator for "win fraction"
        for r in rs:
            ran = 0
            won = 0
            for nom_attr, win_attr in (("dd_nominated", "dd_won"),
                                       ("occ_nominated", "occ_won"),
                                       ("dl_nominated", "dl_won")):
                n, w = getattr(r, nom_attr), getattr(r, win_attr)
                # a precursor "ran" for this entity if it nominated or named a winner
                if n or w:
                    ran += 1
                    if w:
                        won += 1
            # NYDCC only declares winners
            if r.nydcc_won:
                ran += 1
                won += 1
            r.con_win_fraction = (won / ran) if ran else 0.0
            r.con_swept_all = 1.0 if (ran >= 2 and won == ran) else 0.0
        # sole winner: the only nominee in the race that won ANY precursor
        winners = [r for r in rs if r.n_precursor_wins and r.n_precursor_wins > 0]
        if len(winners) == 1:
            winners[0].con_sole_winner = 1.0


# external data is loaded once and cached across calls
_EXT_CACHE = None
_TR_CACHE = None


def _assign_transfers(season, rows):
    """Attach transfer features: credit for an Olivier win on the same West End
    production/role, or major Off-Broadway acclaim in a prior season before a
    Broadway transfer. Addresses the blind spot that the base model only sees
    same-season precursors. Time-locked (Oliviers predate the Tonys); see
    transfers.py. Neutral defaults when a show didn't transfer (the common case)."""
    global _TR_CACHE
    if _TR_CACHE is None:
        import json, os
        path = os.path.join(os.path.dirname(__file__), "..", "data",
                            "transfers.json")
        _TR_CACHE = {}
        if os.path.exists(path):
            with open(path) as f:
                for s in json.load(f):
                    _TR_CACHE[s["year"]] = {norm(x["show"]): x
                                            for x in s.get("shows", [])}
    ymap = _TR_CACHE.get(season["year"], {})
    for r in rows:
        info = ymap.get(norm(r.nominee_show), {})
        r.tr_olivier = 1.0 if info.get("won_olivier") is True else 0.0
        r.tr_offbway_acclaim = 1.0 if info.get("prior_offbroadway_acclaim") is True else 0.0
        tf = info.get("transferred_from", "none")
        r.tr_transferred = 1.0 if tf in ("london", "offbroadway", "both") else 0.0


def _assign_external(season, rows):
    """Attach time-locked external per-show features (commercial/critical/pundit)
    to each row. Defaults are neutral when a show is absent from the external
    dataset, so the model degrades gracefully. See external.py for the analysis.
    All external data is recorded as-of Tony ceremony day only (leak-guarded)."""
    global _EXT_CACHE
    if _EXT_CACHE is None:
        import json, os
        path = os.path.join(os.path.dirname(__file__), "..", "data",
                            "external_shows.json")
        _EXT_CACHE = {}
        if os.path.exists(path):
            with open(path) as f:
                for s in json.load(f):
                    _EXT_CACHE[s["year"]] = {norm(x["show"]): x
                                             for x in s.get("shows", [])}
    ymap = _EXT_CACHE.get(season["year"], {})
    for r in rows:
        info = ymap.get(norm(r.nominee_show), {})
        r.ext_open = _extbool(info.get("still_open_at_tonys"), 1.0)
        r.ext_closed = _extbool(info.get("closed_before_tonys"), 0.0)
        r.ext_pundit_fav = _extbool(info.get("is_pundit_favorite"), 0.0)
        rec = info.get("critics_reception", "unknown")
        r.ext_rave = 1.0 if rec == "rave" else 0.0
        r.ext_wellreviewed = 1.0 if rec in ("rave", "positive") else 0.0
        run = info.get("run_longevity", "unknown")
        r.ext_hit = 1.0 if run == "hit_long_run" else 0.0
        r.ext_flop = 1.0 if run == "short_flop" else 0.0


def _extbool(v, default):
    if v is True:
        return 1.0
    if v is False:
        return 0.0
    return default


def _assign_frontrunner(season, rows):
    """Compute the 'sweep' signal: how many Tony categories each SHOW is
    nominated in this season, then flag the nominee(s) whose show uniquely leads
    its own category race. Available pre-ceremony (nom counts are public), so
    leak-free. Backtested at +2.4 pts overall (experiments.py)."""
    show_counts = {}
    for cat, data in season["tony"].items():
        seen = set()
        for n in data["nominees"]:
            s = norm(n["show"])
            if s not in seen:        # one show, two noms in a category = +1
                show_counts[s] = show_counts.get(s, 0) + 1
                seen.add(s)
    by_cat = {}
    for r in rows:
        r.show_nom_count = float(show_counts.get(norm(r.nominee_show), 0))
        by_cat.setdefault(r.category, []).append(r)
    for cat, rs in by_cat.items():
        mx = max((r.show_nom_count for r in rs), default=0.0)
        leaders = [r for r in rs if r.show_nom_count == mx and mx > 0]
        # only a meaningful signal when ONE show leads (not all tied)
        if mx > 0 and len(leaders) < len(rs):
            for r in leaders:
                r.is_frontrunner = 1.0


# --- small helpers that tolerate dicts loaded from JSON ---------------------
def _nominee_key(n: dict) -> str:
    if n.get("person"):
        return f"{norm(n['person'])}|{norm(n['show'])}"
    return norm(n["show"])


def _result_key(r: dict) -> str:
    if r.get("person"):
        return f"{norm(r['person'])}|{norm(r['show'])}"
    return norm(r["show"])


def _result_is_musical(r: dict) -> bool:
    """Best-effort: a combined design result carries a 'show_type' hint
    ('musical'/'play') when known; default to matching neither strictly by
    returning the explicit flag."""
    return (r.get("show_type") == "musical")


def _assign_signal(fr: FeatureRow, precursor: str, nominated: bool, won: bool):
    nf = 1.0 if nominated else 0.0
    wf = 1.0 if won else 0.0
    if precursor == "drama_desk":
        fr.dd_nominated, fr.dd_won = nf, wf
    elif precursor == "occ":
        fr.occ_nominated, fr.occ_won = nf, wf
    elif precursor == "drama_league":
        fr.dl_nominated, fr.dl_won = nf, wf
    elif precursor == "nydcc":
        fr.nydcc_won = wf  # NYDCC declares only winners


def _finalize(fr: FeatureRow):
    wins = [fr.dd_won, fr.occ_won, fr.dl_won, fr.nydcc_won]
    noms = [fr.dd_nominated, fr.occ_nominated, fr.dl_nominated]
    fr.n_precursor_wins = sum(w for w in wins if w)
    fr.n_precursor_noms = sum(n for n in noms if n)


if __name__ == "__main__":
    # minimal smoke test with a fabricated mini-season
    season = {
        "year": 2024,
        "tony": {
            "best_play": {"nominees": [
                {"show": "Stereophonic", "won": True},
                {"show": "Mary Jane", "won": False},
            ], "winner": "stereophonic"},
        },
        "precursors": {},
    }
    # pad to full category set so build_features doesn't KeyError
    from categories import TONY_CATEGORIES
    for cat in TONY_CATEGORIES:
        season["tony"].setdefault(cat, {"nominees": [], "winner": None})
    precursors = {
        "drama_desk": [{"category": "Outstanding Play", "show": "Stereophonic",
                        "won": True, "is_broadway": True}],
        "occ": [{"category": "Outstanding New Broadway Play", "show": "Stereophonic",
                 "won": True, "is_broadway": True}],
    }
    rows = build_features(season, precursors)
    win_row = next(r for r in rows if r.nominee_show == "Stereophonic")
    assert win_row.dd_won == 1.0 and win_row.occ_won == 1.0, win_row
    assert win_row.n_precursor_wins == 2.0
    print("OK: matcher smoke test passes.")
