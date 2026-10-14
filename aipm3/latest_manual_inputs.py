"""Source-bound manual observations; no changes to frozen model artifacts."""
from copy import deepcopy

LEGACY_CELEBRITY_VERSION = "latest-manual-celebrity-v1"
CELEBRITY_VERSION = "latest-manual-celebrity-v2"
RECALL_MULTIPLIER = 1.2
NOTICEABILITY_MULTIPLIER = 1.3


def celebrity_presence(metadata):
    review = metadata.get("celebrity_review")
    if review is None:
        return None
    if (not isinstance(review, dict) or review.get("version") not in {
            CELEBRITY_VERSION, LEGACY_CELEBRITY_VERSION}
            or type(review.get("present")) is not bool
            or not metadata.get("source_sha")
            or review.get("source_sha") != metadata.get("source_sha")):
        raise ValueError("Не удалось проверить отметку о медийной персоне для этого ролика.")
    return review["present"]


def effective_feature_rows(feature_rows, metadata):
    rows = deepcopy(feature_rows)
    present = celebrity_presence(metadata)
    if present is not None:
        for row in rows["n"]:
            row["is_celeb"] = int(present)
    return rows


def recall_multiplier(metadata):
    return RECALL_MULTIPLIER if celebrity_presence(metadata) is True else 1.0


def noticeability_multiplier(metadata):
    present = celebrity_presence(metadata)
    # Validate old exports under their original policy before upgrading them.
    current = (metadata.get("celebrity_review") or {}).get("version") == CELEBRITY_VERSION
    return NOTICEABILITY_MULTIPLIER if present is True and current else 1.0


def with_celebrity_review(result, present):
    """Recalculate from preserved observations, never from adjusted scores."""
    from .latest_runtime import result_metadata, score_feature_rows, material_kind_for_result
    if type(present) is not bool:
        raise ValueError("Отметка о медийной персоне должна быть да или нет.")
    metadata = result_metadata(result)
    metadata["celebrity_review"] = dict(
        version=CELEBRITY_VERSION, present=present,
        source_sha=result.get("source_sha"), origin="user_declared")
    return score_feature_rows(result["feature_rows"], metadata=metadata,
                              material_kind=material_kind_for_result(result),
                              repeat_count=result.get("repeat_count", 10))
