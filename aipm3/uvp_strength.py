"""Evidence-aware aggregation of five blind UVP ratings. No campaign-specific parameters."""
from statistics import median


MODEL = "uvp-evidence-four-of-five-v1"
REQUIRED_SUPPORT = 4
# A literalism flag often means a judge has UNDERRATED a metaphor.
# It remains in the report, but does not invalidate support for the observed level.
UNUSABLE_ARGUMENT = frozenset({
    "fabricated_fact", "scope_error", "contradiction", "clarity_strength_conflation",
    "unsupported_negative", "invented_audience",
})


def signed_strength(row):
    magnitude = row["strength"]
    direction = row["direction"]
    if type(magnitude) is not int or magnitude not in range(4):
        raise ValueError("Strength outside 0..3")
    if direction not in ("positive", "negative", "neutral") or (magnitude == 0) != (direction == "neutral"):
        raise ValueError("Direction is inconsistent with strength")
    return -magnitude if direction == "negative" else magnitude


def corroborated(votes, usable):
    """Require four evidenced votes at least as strong and in the same direction."""
    if len(votes) != 5 or len(usable) != 5:
        raise ValueError("Exactly five independent judgements are required")
    original = int(median(votes))
    if original == 0:
        return 0, sum(v == 0 and ok for v, ok in zip(votes, usable))
    sign = 1 if original > 0 else -1
    supporters = sum(ok and v * sign >= abs(original) for v, ok in zip(votes, usable))
    return (original if supporters >= REQUIRED_SUPPORT else original - sign), supporters


def score_image(rows, issues, source_ids):
    votes = [signed_strength(r) for r in rows]
    if {r["id"] for r in rows} != {rows[0]["id"]}:
        raise ValueError("Rows must assess the same image")
    if any(i["image_id"] != rows[0]["id"] or i["model_index"] not in range(5) for i in issues):
        raise ValueError("Checker flag assigned to a different image or judge")
    faulty = {i["model_index"] for i in issues if i["type"] in UNUSABLE_ARGUMENT}
    usable = []
    for i, row in enumerate(rows):
        cited = {sid for e in row["evidence"] if e["observation"].strip() for sid in e["source_ids"]}
        if cited - source_ids:
            raise ValueError("Unknown evidence reference")
        usable.append(i not in faulty and (bool(cited) or votes[i] == 0))
    score, qualified = corroborated(votes, usable)
    return dict(before=int(median(votes)), after=score, votes=votes, usable=usable, qualified=qualified,
                clarity=int(median(r["clarity"] for r in rows)),
                review=max(votes) - min(votes) >= 2 or bool(issues), reasons=issues)
