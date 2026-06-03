"""
Vote-splitting heuristic (EXPLICIT rule, not learned).

The sharpest critique from readers: the model scores each nominee independently,
so it can't see two SIMILAR shows splitting a shared voter bloc and handing the
win to a third. The canonical 2026 example: Schmigadoon! and Titanique are both
broad comedy/parody musicals; if they split the "fun musical" vote, a more
earnest show (The Lost Boys) can come up the middle.

WHY THIS IS A HEURISTIC, NOT A LEARNED FEATURE: vote-splitting is real but rare,
and with ~16-25 Best Musical races there are nowhere near enough labeled split
examples to learn the effect without overfitting. So this is a transparent,
hand-specified rule: when two nominees in a category share a "lineage tag", we
shave a fixed fraction off each of their scores and redistribute it. The tags
and the penalty are declared here in the open - it is an editorial judgment, and
it is applied only at predict time as an optional adjustment, never baked into
the backtested model accuracy.

Because there's no way to validate it against history (too few cases), we DO NOT
claim it improves accuracy. It's offered as an optional "what if there's a
split?" lens on a prediction.
"""
from schema import norm

# Lineage tags: nominees whose shows compete for the same voter bloc. Keyed by
# the season year, then a tag -> list of show titles. Hand-curated and dated so
# the rule is auditable. Only the 2026 race is tagged for now (the live one);
# historical tagging could be added but isn't used in the backtest.
LINEAGE_TAGS = {
    2026: {
        # broad comedy / parody / "pure fun" musicals competing for the same
        # not-the-earnest-drama vote in Best Musical
        "comedy_parody_musical": [
            "Schmigadoon!",
            "Titanique",
        ],
    },
}

# Fraction of each tied show's score to move to the rest of the field when a
# split is detected. 0.15 is deliberately modest - this is a nudge, not a verdict.
SPLIT_PENALTY = 0.15


def apply_vote_split(year, category, ranked):
    """ranked: list of (row, prob, raw) from predict_category (already calibrated
    or not). Returns a re-weighted, re-normalized list reflecting a possible
    split among same-lineage shows. No-op when fewer than 2 tagged shows are
    actually nominated in this race."""
    tags = LINEAGE_TAGS.get(year, {})
    if not tags or not ranked:
        return ranked
    show_of = {id(r): norm(r.nominee_show) for r, _, _ in ranked}
    # which tag groups have >=2 nominees present in this race?
    penalized = set()
    for tag, shows in tags.items():
        tagged_norm = {norm(s) for s in shows}
        present = [r for r, _, _ in ranked if show_of[id(r)] in tagged_norm]
        if len(present) >= 2:
            penalized.update(id(r) for r in present)
    if not penalized:
        return ranked

    # shave SPLIT_PENALTY off each penalized nominee's prob; redistribute the
    # freed mass proportionally across the non-penalized nominees.
    freed = 0.0
    adj = []
    for r, p, raw in ranked:
        if id(r) in penalized:
            keep = p * (1 - SPLIT_PENALTY)
            freed += p - keep
            adj.append([r, keep, raw])
        else:
            adj.append([r, p, raw])
    others = [a for a in adj if id(a[0]) not in penalized]
    if others:
        denom = sum(a[1] for a in others) or 1.0
        for a in others:
            a[1] += freed * (a[1] / denom)
    else:
        # everyone was penalized (whole field same lineage) - just renormalize
        pass
    total = sum(a[1] for a in adj) or 1.0
    out = [(a[0], a[1] / total, a[2]) for a in adj]
    out.sort(key=lambda t: (-t[1], t[0].nominee_key))
    return out


if __name__ == "__main__":
    from collections import defaultdict
    from validate import load_all_seasons, season_to_rows
    from model import NaiveBayesTonyModel
    seasons = load_all_seasons()
    rby = {s["year"]: season_to_rows(s) for s in seasons}
    train = [r for y, rows in rby.items() if y != 2026 for r in rows if r.won_tony is not None]
    m = NaiveBayesTonyModel().fit(train)
    target = next(s for s in seasons if s["year"] == 2026)
    bc = defaultdict(list)
    for r in season_to_rows(target):
        bc[r.category].append(r)
    ranked = m.predict_category(bc["best_musical"])
    print("Best Musical 2026 - WITHOUT vote-split adjustment:")
    for r, p, _ in ranked:
        print(f"  {p*100:4.0f}%  {r.nominee_show}")
    print("\nWITH vote-split adjustment (Schmigadoon!/Titanique share a bloc):")
    for r, p, _ in apply_vote_split(2026, "best_musical", ranked):
        print(f"  {p*100:4.0f}%  {r.nominee_show}")
