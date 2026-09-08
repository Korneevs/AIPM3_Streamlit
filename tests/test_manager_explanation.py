from copy import deepcopy

import pytest

from aipm3.manager_explanation import (
    MANAGER_EXPLANATION_VERSION,
    build_main_message_evidence,
    build_manager_explanation,
)


def row(uid, idea="Первый посыл", kind="valid", condition="full", answer=None):
    return {
        "respondent_uid": uid,
        "condition_group": condition,
        "canonical_idea": idea,
        "answer_type": kind,
        "raw_answer": answer or f"Пересказ {uid}",
    }


def result(rows=None, candidate="Первый посыл", percentiles=(55.0, 38.0, 7.0)):
    return {
        "aipm3": {"index_100": 54.0, "label": "Ниже среднего"},
        "aipm1": {"percentile": percentiles[0]},
        "aipm2": {"percentile": percentiles[1]},
        "message_delivery": {"percentile": percentiles[2]},
        "main_idea": candidate,
        "diagnostic_recovery": [] if rows is None else rows,
    }


def test_noticeability_observation_uses_prompt_category_not_measured_seconds():
    data = result(percentiles=(19, 46, 34))
    data['aipm1']['feature_values'] = {'message_focus_seconds': 2}
    before = deepcopy(data)
    explanation = build_manager_explanation(data)
    assert '«с усилием»' in explanation['components'][0]['source_observation']
    assert '«с усилием»' in explanation['summary']
    assert 'секунд' not in explanation['components'][0]['source_observation']
    assert data == before
    for unavailable in [None, 4, [], True]:
        data['aipm1']['feature_values']['message_focus_seconds'] = unavailable
        assert 'source_observation' not in build_manager_explanation(data)['components'][0]


def test_full_cohort_counts_include_no_idea_and_brand_only_but_not_other_panels():
    data = result([
        row("f1"), row("f2"), row("f3", "Другой посыл"),
        row("f4", "<BRAND_ONLY>", "brand_only"), row("f5", "<NO_IDEA>", "no_idea"),
        *[row(f"masked{i}", condition="nested_25_m1") for i in range(20)],
    ])
    data["diagnostic_panel"] = [{"main_message_summary": "Первый посыл"} for _ in range(30)]
    evidence = build_main_message_evidence(data)
    assert evidence["status"] == "available"
    assert evidence["full_count"] == evidence["support_denominator"] == 5
    assert evidence["valid_count"] == 3
    assert evidence["brand_only_count"] == evidence["no_idea_count"] == 1
    assert evidence["support_count"] == 2
    assert evidence["majority"] is False
    assert "2 из 5" in evidence["summary"]


def test_tied_clusters_are_both_visible_and_do_not_claim_one_main_message():
    evidence = build_main_message_evidence(result([
        row("a1"), row("b1", "Другой посыл"), row("a2"), row("b2", "Другой посыл"),
        row("c", "<BRAND_ONLY>", "brand_only"),
    ]))
    assert set(evidence["top_ideas"]) == {"Первый посыл", "Другой посыл"}
    assert evidence["majority"] is False
    assert evidence["dominant_idea"] is None
    assert "по 2 из 5" in evidence["summary"]
    assert "Первый посыл" in evidence["summary"] and "Другой посыл" in evidence["summary"]


@pytest.mark.parametrize("support,total,majority", [(3, 6, False), (4, 6, True), (7, 13, True)])
def test_majority_means_strictly_more_than_half_of_all_observed_full_answers(support, total, majority):
    rows = [row(f"v{i}") for i in range(support)]
    rows += [row(f"n{i}", "<NO_IDEA>", "no_idea") for i in range(total - support)]
    evidence = build_main_message_evidence(result(rows))
    assert evidence["full_count"] == total
    assert evidence["majority"] is majority


def test_displayed_candidate_not_in_answers_has_zero_support_without_becoming_top():
    evidence = build_main_message_evidence(result([row("a", "Другой посыл"), row("b", "Другой посыл")]))
    assert evidence["support_count"] == 0
    assert evidence["dominant_idea"] == "Другой посыл"
    assert "0 из 2" in evidence["summary"]


def test_no_valid_message_is_not_a_new_canonical_cluster():
    evidence = build_main_message_evidence(result([
        row("b", "<BRAND_ONLY>", "brand_only"), row("n", "<NO_IDEA>", "no_idea"),
    ], candidate="<NO_IDEA>"))
    assert evidence["clusters"] == []
    assert evidence["top_ideas"] == []
    assert evidence["support_count"] is None
    assert evidence["displayed_candidate"] is None
    assert evidence["majority"] is False


@pytest.mark.parametrize("recovery", [None, [], {}, "broken", [row("m", condition="nested_25_m1")]])
def test_missing_full_diagnostics_remains_unavailable(recovery):
    data = result()
    data["diagnostic_recovery"] = recovery
    evidence = build_main_message_evidence(data)
    assert evidence["status"] == "unavailable"
    assert evidence["support_count"] is None
    assert evidence["majority"] is None


def test_partial_classification_is_not_counted_as_consensus_or_missing_idea():
    evidence = build_main_message_evidence(result([
        row("a"), row("b", "", "valid"), row("c", "<NO_IDEA>", "valid"),
        row("d", "Обычная фраза", "brand_only"),
    ]))
    assert evidence["status"] == "partial"
    assert evidence["full_count"] == 4
    assert evidence["classified_count"] == 1
    assert evidence["unclassified_count"] == 3
    assert evidence["no_idea_count"] == evidence["brand_only_count"] == 0
    assert evidence["support_count"] == 1
    assert evidence["majority"] is None


def test_duplicate_identity_excludes_both_conflicting_rows_and_preserves_observed_denominator():
    evidence = build_main_message_evidence(result([
        row("same"), row("same", "Другой посыл"), row("unique"),
    ]))
    assert evidence["status"] == "partial"
    assert evidence["full_count"] == 3
    assert evidence["duplicate_row_count"] == 2
    assert evidence["classified_count"] == evidence["support_count"] == 1
    assert evidence["unclassified_count"] == 2
    assert evidence["majority"] is None


def test_malformed_unassigned_and_missing_identity_rows_surface_incompleteness():
    evidence = build_main_message_evidence(result([row(""), {"canonical_idea": "Первый посыл"}, None]))
    assert evidence["status"] == "partial"
    assert evidence["unassigned_count"] == 2
    assert evidence["full_count"] == 1
    assert evidence["majority"] is None
    assert len(evidence["issues"]) == 2


def test_examples_keep_at_most_two_distinct_raw_answers_and_no_persona_fields():
    rows = [row(f"u{i}", answer=f"Пересказ {i % 3}") for i in range(5)]
    rows[0]["persona"] = "Synthetic demographic description"
    evidence = build_main_message_evidence(result(rows))
    assert evidence["clusters"][0]["examples"] == ["Пересказ 0", "Пересказ 1"]
    assert "persona" not in repr(evidence)


def test_summary_prioritizes_investigation_and_keeps_causal_limits_in_uncertainty_notes():
    data = result([row("a")])
    before = deepcopy(data)
    explanation = build_manager_explanation(data)
    assert data == before
    assert explanation["version"] == MANAGER_EXPLANATION_VERSION
    assert explanation["overall"]["index_100"] == 54.0
    assert explanation["weakest_keys"] == ["message_delivery"]
    assert explanation["strongest_keys"] == ["aipm1"]
    assert "Это первый компонент для дополнительной проверки" in explanation["summary"]
    assert "причины итогового балла" not in explanation["summary"]
    assert any("не доказывает причины итогового балла" in note for note in explanation["uncertainty_notes"])
    assert "AIPM 3.0" not in explanation["summary"]
    assert all(component["inspect_question"].endswith("?") for component in explanation["components"])
    assert "100 — средний" in explanation["overall"]["scale_note"]
    assert all("Позиция в исторической базе" in component["position"] for component in explanation["components"])
    assert "процентиль" not in repr(explanation).lower()
    assert explanation["components"][2]["outcome"] == (
        "По прогнозу модели, считываемость ролика низкая относительно исторической базы."
    )


@pytest.mark.parametrize("percentile,level", [(32.999, "Низкий"), (33.0, "Средний"), (66.999, "Средний"), (67.0, "Высокий")])
def test_existing_component_bands_are_unchanged(percentile, level):
    explanation = build_manager_explanation(result(percentiles=(percentile,) * 3))
    assert all(component["level"] == level for component in explanation["components"])


def test_rank_ties_are_preserved_including_all_equal_components():
    explanation = build_manager_explanation(result(percentiles=(20, 20, 70)))
    assert explanation["weakest_keys"] == ["aipm1", "aipm2"]
    assert "Эти компоненты стоит проверить в первую очередь" in explanation["summary"]
    equal = build_manager_explanation(result(percentiles=(50, 50, 50)))
    assert equal["weakest_keys"] == equal["strongest_keys"] == ["aipm1", "aipm2", "message_delivery"]
    assert "совпадают" in equal["summary"]


def test_missing_nonfinite_and_out_of_range_scores_are_never_classified_as_high():
    explanation = build_manager_explanation(result(percentiles=(None, float("nan"), 101)))
    assert explanation["weakest_keys"] == explanation["strongest_keys"] == []
    assert all(component["level"] is None for component in explanation["components"])
    assert "недоступно" in explanation["summary"]


def test_saved_fractional_profile_values_explain_discrepancy_without_rounding_to_integer():
    data = result()
    data["feature_profile"] = {"measurements": [
        {"feature": "audio_only_message_completeness", "value": 2.76666666667},
        {"feature": "visual_only_message_completeness", "value": 1.96666666667},
    ]}
    data["message_delivery_business"] = {"audio_completeness": 3}
    explanation = build_manager_explanation(data)
    detail = explanation["discrepancies"][0]["detail"]
    assert "2,8/3" in detail and "2/3" in detail
    assert "не означает" in detail
