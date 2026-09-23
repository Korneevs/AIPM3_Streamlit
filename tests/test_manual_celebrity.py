"""Manual participation adjusts the index without changing frozen predictions."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from aipm3 import models
from aipm3.manual_celebrity import (
    CELEBRITIES, VERSION, apply_celebrity, selected_celebrity,
)
from aipm3.result_export import export_result
from aipm3.summary_ui import metric_summaries
from test_summary_ui import sample_result


@pytest.fixture
def original():
    result = sample_result()
    result["source_sha"] = "manual-celebrity-test-video"
    result["aipm2"].update(
        raw_score=0.123456789,
        reference_score=0.1235,
        feature_values={"brand_logo_screen_seconds": 4},
    )
    result["aipm3"] = models.aipm3_score(
        result["aipm1"], result["aipm2"], result["message_delivery"],
    )
    result["vertical_uvp"] = {
        "target": {"vertical": "Товары", "goods": "Ресейл"},
        "status": "absent", "counts": {"matched": 0, "absent": 30},
    }
    result["blind_answers"] = [{"respondent_id": "1", "answer": "Продать вещь"}]
    return result


def assert_frozen_evidence_unchanged(actual, expected):
    for component in ("aipm1", "message_delivery"):
        assert actual[component] == expected[component]
    for field in ("raw_score", "reference_score", "percentile", "feature_values", "feature_effects"):
        assert actual["aipm2"][field] == expected["aipm2"][field]
    for field in ("scoring_version", "model_sha256", "source_sha", "vertical_uvp", "blind_answers"):
        assert actual[field] == expected[field]


def test_choice_contract_and_legacy_result_default(original):
    assert set(CELEBRITIES) == {"none", "fomenko", "zhuravlyov", "kurkova"}
    assert VERSION == "manual-celebrity-v1"
    assert selected_celebrity(original) == "none"


def test_none_preserves_all_original_scores_exactly(original):
    before = deepcopy(original)
    adjusted = apply_celebrity(original, "none")
    assert original == before
    assert adjusted is not original
    for component in ("aipm1", "aipm2", "message_delivery", "aipm3"):
        assert adjusted[component] == original[component]
    info = adjusted["celebrity_adjustment"]
    assert info["present"] == 0
    assert info["multiplier"] == 1
    assert info["final_share_percent"] == 0
    assert selected_celebrity(adjusted) == "none"


@pytest.mark.parametrize("selection", ["fomenko", "zhuravlyov", "kurkova"])
def test_each_manual_celebrity_adjusts_only_recall_and_overall(original, selection):
    before = deepcopy(original)
    adjusted = apply_celebrity(original, selection)
    assert original == before
    assert_frozen_evidence_unchanged(adjusted, original)
    assert adjusted["aipm2"]["reference_index"] == original["aipm2"]["reference_index"] * 1.3
    assert adjusted["aipm3"]["index"] == pytest.approx(original["aipm3"]["index"] * 1.3)
    assert adjusted["aipm3"]["index_100"] == pytest.approx(original["aipm3"]["index_100"] * 1.3)
    assert adjusted["aipm3"]["component_indices"]["aipm2"] == adjusted["aipm2"]["reference_index"]
    info = adjusted["celebrity_adjustment"]
    assert info["version"] == VERSION
    assert info["source"] == "manual"
    assert info["source_sha"] == original["source_sha"]
    assert info["component"] == "aipm2"
    assert info["selection"] == selected_celebrity(adjusted) == selection
    assert info["present"] == 1
    assert info["multiplier"] == 1.3
    assert info["base_aipm2"] == original["aipm2"]
    assert info["base_aipm3"] == original["aipm3"]
    assert info["final_share_percent"] == pytest.approx(100 * 0.3 / 1.3)


def test_repeated_application_and_name_changes_never_compound(original):
    first = apply_celebrity(original, "fomenko")
    assert apply_celebrity(first, "fomenko") == first
    adjusted = first
    for selection in ("zhuravlyov", "kurkova", "fomenko", "none", "kurkova"):
        before = deepcopy(adjusted)
        adjusted = apply_celebrity(adjusted, selection)
        assert before["celebrity_adjustment"]["base_aipm2"] == original["aipm2"]
        assert adjusted == apply_celebrity(original, selection)
    restored = apply_celebrity(adjusted, "none")
    assert restored == apply_celebrity(original, "none")
    assert restored["aipm2"] == original["aipm2"]
    assert restored["aipm3"] == original["aipm3"]
    assert "norm_level" not in restored["aipm2"]


def test_snapshots_and_export_do_not_alias_source(original):
    before = deepcopy(original)
    adjusted = apply_celebrity(original, "fomenko")
    exported = export_result(adjusted)
    assert original == before
    assert exported["celebrity_adjustment"] == adjusted["celebrity_adjustment"]
    serialized = json.loads(json.dumps(exported))
    assert apply_celebrity(serialized, "fomenko") == exported
    restored = apply_celebrity(serialized, "none")
    assert restored["aipm2"] == original["aipm2"]
    assert restored["aipm3"] == original["aipm3"]
    exported["celebrity_adjustment"]["base_aipm2"]["feature_values"].clear()
    exported["vertical_uvp"]["counts"]["matched"] = 99
    assert adjusted["celebrity_adjustment"]["base_aipm2"] == original["aipm2"]
    assert adjusted["vertical_uvp"] == original["vertical_uvp"]
    adjusted["aipm2"]["feature_values"]["brand_logo_screen_seconds"] = 99
    assert adjusted["celebrity_adjustment"]["base_aipm2"] == original["aipm2"]
    assert original == before


@pytest.mark.parametrize("effective_index,level", [
    (0.8299, 0), (0.83, 1), (1.1699, 1), (1.17, 2), (1.95, 2),
])
def test_summary_uses_adjusted_norm_level_without_rewriting_percentile(original, effective_index, level):
    original["aipm2"]["reference_index"] = effective_index / 1.3
    original["aipm2"]["percentile"] = 100 * (effective_index / 1.3 - 0.5)
    adjusted = apply_celebrity(original, "fomenko")
    assert adjusted["aipm2"]["norm_level"] == level
    assert metric_summaries(adjusted)[2]["level"] == level
    assert adjusted["aipm2"]["percentile"] == original["aipm2"]["percentile"]
    assert adjusted["aipm2"]["reference_index"] == pytest.approx(effective_index)


def test_largest_reference_index_is_not_capped_after_adjustment(original):
    original["aipm2"]["reference_index"] = 1.5
    adjusted = apply_celebrity(original, "kurkova")
    assert adjusted["aipm2"]["reference_index"] == pytest.approx(1.95)
    assert adjusted["aipm3"]["component_indices"]["aipm2"] == pytest.approx(1.95)
    assert metric_summaries(adjusted)[2]["level"] == 2


@pytest.mark.parametrize("selection", ["someone_else", "", None, 1])
def test_unknown_selection_is_rejected(original, selection):
    with pytest.raises(ValueError):
        apply_celebrity(original, selection)


def test_unknown_policy_and_wrong_video_cannot_reuse_saved_adjustment(original):
    adjusted = apply_celebrity(original, "fomenko")
    unsupported = deepcopy(adjusted)
    unsupported["celebrity_adjustment"]["version"] = "future-unknown-policy"
    with pytest.raises(ValueError):
        apply_celebrity(unsupported, "none")
    another_video = deepcopy(adjusted)
    another_video["source_sha"] = "different-video"
    with pytest.raises(ValueError):
        apply_celebrity(another_video, "kurkova")


@pytest.mark.parametrize("value", [0, -0.1, float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("saved_snapshot", [False, True])
def test_invalid_base_index_cannot_produce_adjusted_scores(original, value, saved_snapshot):
    result = apply_celebrity(original, "fomenko") if saved_snapshot else deepcopy(original)
    base = result["celebrity_adjustment"]["base_aipm2"] if saved_snapshot else result["aipm2"]
    base["reference_index"] = value
    with pytest.raises(ValueError):
        apply_celebrity(result, "fomenko")


@pytest.mark.parametrize("filename", ["resale_result.json", "sale_result.json"])
def test_saved_real_video_results_replay_without_extraction_or_api(monkeypatch, filename):
    path = Path.home() / "outputs/aipm3_uvp_validation_20260923" / filename
    if not path.exists():
        pytest.skip("Private saved video results are local-only")
    import openai

    def forbidden(*args, **kwargs):
        pytest.fail("Changing manual participation must not call API or frozen models")

    monkeypatch.setattr(openai, "OpenAI", forbidden)
    for name in ("score_aipm1", "score_aipm2", "score_message_delivery"):
        monkeypatch.setattr(models, name, forbidden)
    result = json.loads(path.read_text())
    before = deepcopy(result)
    adjusted = apply_celebrity(result, "zhuravlyov")
    assert result == before
    assert adjusted["aipm2"]["reference_index"] == result["aipm2"]["reference_index"] * 1.3
    assert adjusted["aipm3"]["index"] == pytest.approx(result["aipm3"]["index"] * 1.3)
    restored = apply_celebrity(adjusted, "none")
    for component in ("aipm1", "aipm2", "message_delivery", "aipm3"):
        assert restored[component] == result[component]
    if "vertical_uvp" in result:
        assert adjusted["vertical_uvp"] == restored["vertical_uvp"] == result["vertical_uvp"]
