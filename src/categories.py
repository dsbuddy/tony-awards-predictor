"""
Tony Award categories and the crosswalk mapping each to its precursor-award
equivalents.

Why this file matters
---------------------
The single hardest part of predicting the Tonys from precursor awards is that
the precursors do NOT use the same category structure as the Tonys:

  * Drama Desk (DD) often combines Play + Musical into one category that the
    Tonys split (e.g. one DD "Outstanding Lighting Design" historically, vs.
    Tony's separate Play/Musical lighting awards). DD has changed this over
    time, so the crosswalk is approximate and the matcher must fall back to
    name-matching the actual nominee.
  * Drama Desk SPLITS "Best Score" into "Outstanding Music" and "Outstanding
    Lyrics", whereas the Tony has a single "Best Original Score".
  * Outer Critics Circle (OCC) mixes Broadway AND Off-Broadway in most
    categories, and has its own "John Gassner" (new playwright) award with no
    Tony equivalent.
  * Drama League gives essentially ONE production award + acting/"distinguished
    performance" awards, so it maps to very few Tony categories.
  * NY Drama Critics' Circle (NYDCC) only gives Best Play / Best Musical /
    Best Foreign Play -> maps to only Best Play / Best Musical.

So the crosswalk is intentionally MANY-TO-MANY and per-precursor. The matching
engine (see matcher.py) uses it as a hint, then confirms by matching the actual
nominated production/person, because a precursor "win" only counts as signal if
it went to a Tony-ELIGIBLE (Broadway) nominee.
"""

# ---------------------------------------------------------------------------
# The 26 Tony categories we predict. Keys are stable internal IDs.
# ---------------------------------------------------------------------------

TONY_CATEGORIES = {
    # Production awards
    "best_play":            "Best Play",
    "best_musical":         "Best Musical",
    "best_revival_play":    "Best Revival of a Play",
    "best_revival_musical": "Best Revival of a Musical",
    "best_book":            "Best Book of a Musical",
    "best_score":           "Best Original Score",
    # Lead acting
    "lead_actor_play":      "Best Leading Actor in a Play",
    "lead_actress_play":    "Best Leading Actress in a Play",
    "lead_actor_musical":   "Best Leading Actor in a Musical",
    "lead_actress_musical": "Best Leading Actress in a Musical",
    # Featured acting
    "feat_actor_play":      "Best Featured Actor in a Play",
    "feat_actress_play":    "Best Featured Actress in a Play",
    "feat_actor_musical":   "Best Featured Actor in a Musical",
    "feat_actress_musical": "Best Featured Actress in a Musical",
    # Creative / direction
    "direction_play":       "Best Direction of a Play",
    "direction_musical":    "Best Direction of a Musical",
    "choreography":         "Best Choreography",
    "orchestrations":       "Best Orchestrations",
    # Design - Play
    "scenic_play":          "Best Scenic Design of a Play",
    "costume_play":         "Best Costume Design of a Play",
    "lighting_play":        "Best Lighting Design of a Play",
    "sound_play":           "Best Sound Design of a Play",
    # Design - Musical
    "scenic_musical":       "Best Scenic Design of a Musical",
    "costume_musical":      "Best Costume Design of a Musical",
    "lighting_musical":     "Best Lighting Design of a Musical",
    "sound_musical":        "Best Sound Design of a Musical",
}

# Which categories belong to a "musical" vs a "play". Used to disambiguate
# combined precursor categories: a combined precursor design award maps to the
# Tony Play or Musical variant depending on whether the winning show was a
# play or a musical.
MUSICAL_CATEGORIES = {
    "best_musical", "best_revival_musical", "best_book", "best_score",
    "lead_actor_musical", "lead_actress_musical",
    "feat_actor_musical", "feat_actress_musical",
    "direction_musical", "choreography", "orchestrations",
    "scenic_musical", "costume_musical", "lighting_musical", "sound_musical",
}

# ---------------------------------------------------------------------------
# Crosswalk: Tony category -> per-precursor candidate category name(s).
#
# `combined: True`  => the precursor category mixes Play+Musical (or is a single
#                      award), so a precursor result only maps here if the actual
#                      winning show matches this Tony category's play/musical type.
# `offbway_risk: True` => precursor category routinely includes Off-Broadway,
#                      so results MUST be filtered to Tony-eligible nominees.
# An empty list means the precursor has no equivalent -> contributes no signal.
# ---------------------------------------------------------------------------

CROSSWALK = {
    "best_play": {
        "drama_desk":  ["Outstanding Play"],
        "occ":         ["Outstanding New Broadway Play"],
        "drama_league":["Distinguished Production of a Play",
                        "Outstanding Production of a Broadway or Off-Broadway Play"],
        "nydcc":       ["Best Play", "Best American Play"],
    },
    "best_musical": {
        "drama_desk":  ["Outstanding Musical"],
        "occ":         ["Outstanding New Broadway Musical"],
        "drama_league":["Distinguished Production of a Musical",
                        "Outstanding Production of a Broadway or Off-Broadway Musical"],
        "nydcc":       ["Best Musical"],
    },
    "best_revival_play": {
        # DD/OCC usually have a single "Revival" or split Play/Musical revival.
        "drama_desk":  ["Outstanding Revival of a Play", "Outstanding Revival"],
        "occ":         ["Outstanding Revival of a Play", "Outstanding Revival"],
        "drama_league":[],
        "nydcc":       [],
    },
    "best_revival_musical": {
        "drama_desk":  ["Outstanding Revival of a Musical", "Outstanding Revival"],
        "occ":         ["Outstanding Revival of a Musical", "Outstanding Revival"],
        "drama_league":[],
        "nydcc":       [],
    },
    "best_book": {
        "drama_desk":  ["Outstanding Book of a Musical"],
        "occ":         ["Outstanding Book of a Musical"],
        "drama_league":[],
        "nydcc":       [],
    },
    "best_score": {
        # Tony has ONE score award; DD splits into Music + Lyrics. We map both;
        # the matcher treats a show winning EITHER as partial signal and BOTH as
        # strong signal.
        "drama_desk":  ["Outstanding Music", "Outstanding Lyrics"],
        "occ":         ["Outstanding New Score", "Outstanding Score"],
        "drama_league":[],
        "nydcc":       [],
    },
    # NOTE: Drama Desk switched (~2023) to GENDER-NEUTRAL combined performance
    # categories ("Outstanding Lead Performance in a Play"), often with TIES.
    # We list both the modern combined name AND the legacy gendered name; the
    # matcher relies on PERSON-matching to route a combined-category result to
    # the correct gendered Tony category, so listing the combined name in both
    # the actor and actress entries is correct and not double-counting.
    "lead_actor_play": {
        "drama_desk":  ["Outstanding Lead Performance in a Play", "Outstanding Actor in a Play"],
        "occ":         ["Outstanding Actor in a Play"],
        "drama_league":["Distinguished Performance"],   # single combined award
        "nydcc":       [],
    },
    "lead_actress_play": {
        "drama_desk":  ["Outstanding Lead Performance in a Play", "Outstanding Actress in a Play"],
        "occ":         ["Outstanding Actress in a Play"],
        "drama_league":["Distinguished Performance"],
        "nydcc":       [],
    },
    "lead_actor_musical": {
        "drama_desk":  ["Outstanding Lead Performance in a Musical", "Outstanding Actor in a Musical"],
        "occ":         ["Outstanding Actor in a Musical"],
        "drama_league":["Distinguished Performance"],
        "nydcc":       [],
    },
    "lead_actress_musical": {
        "drama_desk":  ["Outstanding Lead Performance in a Musical", "Outstanding Actress in a Musical"],
        "occ":         ["Outstanding Actress in a Musical"],
        "drama_league":["Distinguished Performance"],
        "nydcc":       [],
    },
    "feat_actor_play": {
        "drama_desk":  ["Outstanding Featured Performance in a Play", "Outstanding Featured Actor in a Play"],
        "occ":         ["Outstanding Featured Actor in a Play"],
        "drama_league":[],
        "nydcc":       [],
    },
    "feat_actress_play": {
        "drama_desk":  ["Outstanding Featured Performance in a Play", "Outstanding Featured Actress in a Play"],
        "occ":         ["Outstanding Featured Actress in a Play"],
        "drama_league":[],
        "nydcc":       [],
    },
    "feat_actor_musical": {
        "drama_desk":  ["Outstanding Featured Performance in a Musical", "Outstanding Featured Actor in a Musical"],
        "occ":         ["Outstanding Featured Actor in a Musical"],
        "drama_league":[],
        "nydcc":       [],
    },
    "feat_actress_musical": {
        "drama_desk":  ["Outstanding Featured Performance in a Musical", "Outstanding Featured Actress in a Musical"],
        "occ":         ["Outstanding Featured Actress in a Musical"],
        "drama_league":[],
        "nydcc":       [],
    },
    "direction_play": {
        "drama_desk":  ["Outstanding Direction of a Play", "Outstanding Director of a Play"],
        "occ":         ["Outstanding Director of a Play", "Outstanding Direction of a Play"],
        "drama_league":[],
        "nydcc":       [],
    },
    "direction_musical": {
        "drama_desk":  ["Outstanding Direction of a Musical", "Outstanding Director of a Musical"],
        "occ":         ["Outstanding Director of a Musical", "Outstanding Direction of a Musical"],
        "drama_league":[],
        "nydcc":       [],
    },
    "choreography": {
        "drama_desk":  ["Outstanding Choreography"],
        "occ":         ["Outstanding Choreography"],
        "drama_league":[],
        "nydcc":       [],
    },
    "orchestrations": {
        "drama_desk":  ["Outstanding Orchestrations"],
        "occ":         ["Outstanding Orchestrations"],   # OCC added this only some years
        "drama_league":[],
        "nydcc":       [],
    },
    # ---- Design categories: DD/OCC frequently use a COMBINED (play+musical)
    #      award. combined=True tells the matcher to route by show type. ----
    # Design: DD/OCC have used BOTH a combined (play+musical) award in older
    # years AND split "...of a Play"/"...of a Musical" awards in recent years.
    # We list the split name first (exact, unambiguous) and keep the combined
    # name with combined=True so older seasons still route by show_type. When a
    # split name matches, combined routing is a no-op (the name already encodes
    # play/musical), so it's safe to carry both.
    "scenic_play": {
        "drama_desk":  {"names": ["Outstanding Scenic Design of a Play", "Outstanding Set Design of a Play",
                                  "Outstanding Set Design", "Outstanding Scenic Design"], "combined": True},
        "occ":         {"names": ["Outstanding Scenic Design (Play)", "Outstanding Scenic Design"], "combined": True},
        "drama_league":[], "nydcc": [],
    },
    "scenic_musical": {
        "drama_desk":  {"names": ["Outstanding Scenic Design of a Musical", "Outstanding Set Design of a Musical",
                                  "Outstanding Set Design", "Outstanding Scenic Design"], "combined": True},
        "occ":         {"names": ["Outstanding Scenic Design (Musical)", "Outstanding Scenic Design"], "combined": True},
        "drama_league":[], "nydcc": [],
    },
    "costume_play": {
        "drama_desk":  {"names": ["Outstanding Costume Design of a Play", "Outstanding Costume Design"], "combined": True},
        "occ":         {"names": ["Outstanding Costume Design (Play)", "Outstanding Costume Design"], "combined": True},
        "drama_league":[], "nydcc": [],
    },
    "costume_musical": {
        "drama_desk":  {"names": ["Outstanding Costume Design of a Musical", "Outstanding Costume Design"], "combined": True},
        "occ":         {"names": ["Outstanding Costume Design (Musical)", "Outstanding Costume Design"], "combined": True},
        "drama_league":[], "nydcc": [],
    },
    "lighting_play": {
        "drama_desk":  {"names": ["Outstanding Lighting Design of a Play", "Outstanding Lighting Design"], "combined": True},
        "occ":         {"names": ["Outstanding Lighting Design (Play)", "Outstanding Lighting Design"], "combined": True},
        "drama_league":[], "nydcc": [],
    },
    "lighting_musical": {
        "drama_desk":  {"names": ["Outstanding Lighting Design of a Musical", "Outstanding Lighting Design"], "combined": True},
        "occ":         {"names": ["Outstanding Lighting Design (Musical)", "Outstanding Lighting Design"], "combined": True},
        "drama_league":[], "nydcc": [],
    },
    "sound_play": {
        "drama_desk":  {"names": ["Outstanding Sound Design of a Play", "Outstanding Sound Design"], "combined": True},
        "occ":         {"names": ["Outstanding Sound Design (Play)", "Outstanding Sound Design"], "combined": True},
        "drama_league":[], "nydcc": [],
    },
    "sound_musical": {
        "drama_desk":  {"names": ["Outstanding Sound Design of a Musical", "Outstanding Sound Design"], "combined": True},
        "occ":         {"names": ["Outstanding Sound Design (Musical)", "Outstanding Sound Design"], "combined": True},
        "drama_league":[], "nydcc": [],
    },
}

# Precursors that routinely include Off-Broadway and therefore need a
# Broadway-eligibility filter applied to their results before use.
OFFBROADWAY_RISK_PRECURSORS = {"drama_desk", "occ", "drama_league"}

PRECURSORS = ["drama_desk", "occ", "drama_league", "nydcc"]


def is_musical_category(cat_id: str) -> bool:
    return cat_id in MUSICAL_CATEGORIES


# How to decide if a precursor result == a Tony nominee, per category:
#   "person"     -> must match the person (acting: a show has many nominees)
#   "show"       -> match on show/production (design teams & production awards
#                   are credited differently across award bodies, and book/
#                   score go to the show)
# Direction is one named person and usually matches cleanly, but co-directors
# differ across bodies, so we match direction on show too.
ACTING_CATEGORIES = {
    "lead_actor_play", "lead_actress_play", "lead_actor_musical", "lead_actress_musical",
    "feat_actor_play", "feat_actress_play", "feat_actor_musical", "feat_actress_musical",
}


def match_mode(cat_id: str) -> str:
    return "person" if cat_id in ACTING_CATEGORIES else "show"


if __name__ == "__main__":
    assert len(TONY_CATEGORIES) == 26, f"expected 26 categories, got {len(TONY_CATEGORIES)}"
    missing = set(TONY_CATEGORIES) - set(CROSSWALK)
    assert not missing, f"crosswalk missing categories: {missing}"
    print(f"OK: {len(TONY_CATEGORIES)} categories, crosswalk complete.")
