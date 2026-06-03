"""
Data schema for the Tony prediction model.

This defines the exact shape of the data we gather. Everything downstream
(matcher, model, validation) depends on these structures, so this is the
contract the research/data-entry phase must fill.

Storage format on disk
-----------------------
We keep raw data as JSON, one file per season:  data/raw/season_<YEAR>.json
where <YEAR> is the Tony ceremony year (e.g. 2025 = the 78th Tonys, honoring
the 2024-25 Broadway season).

A season file looks like:

{
  "year": 2025,
  "tony": {
     "<tony_category_id>": {
         "nominees": [ {Nominee}, ... ],
         "winner": "<entity key of the winning nominee>"   # null if unknown
     },
     ...
  },
  "precursors": {
     "drama_desk": [ {PrecursorResult}, ... ],
     "occ":        [ {PrecursorResult}, ... ],
     "drama_league":[ ... ],
     "nydcc":      [ ... ]
  }
}

Entity matching
---------------
The whole model hinges on deciding "is this precursor result about the same
nominee as this Tony nominee?". To make that robust we record, for every
nominee and every precursor result, BOTH:
  * show      - the production title (normalized)
  * person    - the performer/creative name, where the award is for a person
For production awards person is null; for acting/design awards person is set.
The matcher (matcher.py) normalizes and compares these.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional


# --- normalization helper: used everywhere we compare titles/names ----------
def norm(s: Optional[str]) -> str:
    """Normalize a title or name for matching: lowercase, strip punctuation,
    collapse whitespace, drop a leading 'the'/'a'. Deliberately aggressive so
    'The Outsiders' == 'Outsiders' and 'Maybe Happy Ending' matches across
    sources with stray punctuation."""
    if not s:
        return ""
    s = s.lower().strip()
    for ch in ".,:;!?'\"’“”()[]&-/":
        s = s.replace(ch, " " if ch in "-/&" else "")
    parts = s.split()
    if parts and parts[0] in ("the", "a", "an"):
        parts = parts[1:]
    return " ".join(parts)


@dataclass
class Nominee:
    """One Tony nominee in one category."""
    show: str                      # production title
    person: Optional[str] = None   # performer/creative; None for production awards
    won: bool = False              # did this nominee win the Tony?

    def key(self) -> str:
        """Stable identity within a category: person+show for personal awards,
        show alone for production awards."""
        if self.person:
            return f"{norm(self.person)}|{norm(self.show)}"
        return norm(self.show)


@dataclass
class PrecursorResult:
    """One precursor-award category outcome for one entity (nominee or winner)."""
    category: str                  # the precursor's OWN category name (verbatim)
    show: str
    person: Optional[str] = None
    won: bool = False              # True = won this precursor; False = nominated only
    is_broadway: Optional[bool] = None  # Tony-eligible? None = unknown (treat cautiously)

    def key(self) -> str:
        if self.person:
            return f"{norm(self.person)}|{norm(self.show)}"
        return norm(self.show)


@dataclass
class CategoryResult:
    nominees: list = field(default_factory=list)   # list[Nominee]
    winner: Optional[str] = None                    # Nominee.key() of winner


# ---------------------------------------------------------------------------
# Feature vector produced per (season, category, nominee) for the model.
# These are the SIGNALS the model weighs. Each precursor contributes two bits:
# did it nominate this entity, and did it give this entity the win.
# ---------------------------------------------------------------------------

@dataclass
class FeatureRow:
    year: int
    category: str
    nominee_key: str
    nominee_show: str
    nominee_person: Optional[str]

    # per-precursor signals (1.0 / 0.0 / None-if-precursor-didn't-run-category)
    dd_nominated: Optional[float] = None
    dd_won: Optional[float] = None
    occ_nominated: Optional[float] = None
    occ_won: Optional[float] = None
    dl_nominated: Optional[float] = None
    dl_won: Optional[float] = None
    nydcc_won: Optional[float] = None      # NYDCC only declares winners

    # derived
    n_precursor_wins: float = 0.0          # how many precursors this entity won
    n_precursor_noms: float = 0.0          # how many precursors nominated it
    is_tony_nominee: float = 1.0           # always 1 here (we only score nominees)
    show_nom_count: float = 0.0            # total Tony noms this nominee's SHOW got
    is_frontrunner: float = 0.0            # 1 if this show uniquely leads its race
    # external time-locked features (see external.py)
    ext_open: float = 1.0                  # still running on ceremony day
    ext_closed: float = 0.0                # closed before ceremony (negative)
    ext_pundit_fav: float = 0.0            # pre-ceremony odds favorite
    ext_rave: float = 0.0                  # opening-night critical raves
    ext_wellreviewed: float = 0.0          # positive-or-better reviews
    ext_hit: float = 0.0                   # commercial hit / long run
    ext_flop: float = 0.0                  # short flop (negative)
    # transfer features (see transfers.py): credit a show's pre-Broadway life
    tr_olivier: float = 0.0                # won an Olivier for the same West End production/role
    tr_offbway_acclaim: float = 0.0        # won a major Off-Broadway award in a prior season
    tr_transferred: float = 0.0            # transferred from London or Off-Broadway at all
    # consensus-strength features (see matcher._assign_consensus): graded signals
    # that capture HOW unanimous the precursor agreement was, not just win/no-win
    con_win_fraction: float = 0.0          # fraction of precursors (that ran this cat) this entity won
    con_swept_all: float = 0.0             # won every precursor that ran the category
    con_sole_winner: float = 0.0           # the only nominee in the race to win any precursor
    # per-category historical pundit/odds favorite (see matcher._assign_odds):
    odds_favorite: float = 0.0             # was this nominee the pre-ceremony favorite in its category
    odds_strong_fav: float = 0.0           # ...and the favorite was a STRONG (not slight/tossup) call
    # prior-career / "overdue" signals (acting + directing; see _assign_prior_noms)
    pn_prior_noms: float = 0.0             # count of prior Tony noms (normalized in matcher)
    pn_overdue: float = 0.0                # >=2 prior noms and never won (the overdue narrative)
    pn_prior_winner: float = 0.0           # has won a Tony before (could cut either way)
    # commercial heat + extra choreography precursor (see _assign_grosses)
    heat_hot: float = 0.0                  # near-capacity / top grosser pre-ceremony
    heat_soft: float = 0.0                 # struggling / low capacity (negative)
    astaire_winner: float = 0.0            # won the Astaire/Chita Rivera choreography award

    # label
    won_tony: Optional[float] = None       # 1/0 for historical rows; None for 2026

    def to_dict(self):
        return asdict(self)


def make_season(year: int) -> dict:
    """Return an empty season skeleton with all 26 categories present."""
    from categories import TONY_CATEGORIES, PRECURSORS
    return {
        "year": year,
        "tony": {cat: {"nominees": [], "winner": None} for cat in TONY_CATEGORIES},
        "precursors": {p: [] for p in PRECURSORS},
    }


if __name__ == "__main__":
    # quick self-checks on the normalizer and keys
    assert norm("The Outsiders") == "outsiders"
    assert norm("A Strange Loop!") == "strange loop"
    assert norm("Maybe Happy Ending") == "maybe happy ending"
    n = Nominee(show="Stereophonic", person="Will Brill", won=True)
    assert n.key() == "will brill|stereophonic"
    p = PrecursorResult(category="Outstanding Play", show="Stereophonic", won=True)
    assert p.key() == "stereophonic"
    print("OK: schema self-checks pass.")
