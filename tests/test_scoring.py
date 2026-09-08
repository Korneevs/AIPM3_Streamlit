import numpy as np

from aipm3.models import (
    AIPM1_INDEX_BY_CLASS,
    AIPM2_REFERENCE,
    AIPM3_PRODUCT_MEAN,
    MD_REFERENCE,
    aipm3_score,
    percentile_index,
    aipm2_reference_index,
)


def test_reference_percentile_reproduces_known_rank() -> None:
    assert percentile_index(0.0926, AIPM2_REFERENCE, 49.0) == 0.6938775510204082
    assert percentile_index(float(MD_REFERENCE.min()), MD_REFERENCE, 49.0) == 0.5204081632653061


def test_aipm2_rounds_only_the_reference_input():
    assert aipm2_reference_index(0.092649) == percentile_index(0.0926, AIPM2_REFERENCE, 49.0)
    assert aipm2_reference_index(0.092551) == percentile_index(0.0926, AIPM2_REFERENCE, 49.0)


def test_float_serialization_cannot_double_count_a_tie():
    value = float(MD_REFERENCE.min())
    expected = percentile_index(value, MD_REFERENCE, 49.0)
    assert percentile_index(value + 1e-16, MD_REFERENCE, 49.0) == expected


def test_aipm3_is_normalized_product() -> None:
    a1 = {"reference_index": AIPM1_INDEX_BY_CLASS[1]}
    a2 = {"reference_index": 1.0}
    md = {"reference_index": 1.0}
    result = aipm3_score(a1, a2, md)
    assert np.isclose(result["index"], AIPM1_INDEX_BY_CLASS[1] / AIPM3_PRODUCT_MEAN)


def test_percentile_index_is_bounded() -> None:
    assert percentile_index(-999, MD_REFERENCE, 49.0) >= 0.5
    assert percentile_index(999, MD_REFERENCE, 49.0) <= 1.5
