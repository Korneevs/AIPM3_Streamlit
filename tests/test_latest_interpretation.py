"""Offline numerical explanations and independent-evidence safety contracts.

The real frozen 9/9/7 heads establish numerical truth. Synthetic attributions
are used only to exercise editorial gates irrespective of a particular clip.
No extraction, provider request, or model fitting is performed here.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aipm3 import latest_evidence as evidence_api
from aipm3 import latest_alignment_evidence as alignment_api
from aipm3 import latest_interpretation as interpretation
from aipm3 import latest_runtime as runtime


INPUTS = Path(__file__).resolve().parents[3] / "outputs/AIPM_interpretation_inputs_20261001"


@pytest.fixture(scope="module")
def frozen_rows():
    if not all((INPUTS / f"ready_{task}.csv").exists() for task in "nmr"):
        pytest.skip("Private verified ten-repeat input cache is not installed")
    frames = {task: pd.read_csv(INPUTS / f"ready_{task}.csv", float_precision="round_trip")
              for task in "nmr"}
    common = set.intersection(*(set(frame.record) for frame in frames.values()))
    record = next(record for record in sorted(common)
                  if all(len(frame[frame.record.eq(record)]) == 10 for frame in frames.values()))
    return {task: frame[frame.record.eq(record)].sort_values("repeat").reset_index(drop=True)
            for task, frame in frames.items()}


@pytest.fixture(scope="module")
def frozen_result(frozen_rows):
    return runtime.score_feature_rows(frozen_rows)


def _background(task):
    train = pd.read_csv(runtime.BUNDLE_DIR / "data" / f"fit_{task}.csv",
                        float_precision="round_trip")
    order = [name for name in ("family", "record", "repeat") if name in train]
    return train.sort_values(order).groupby("family", sort=True).nth(0).reset_index(drop=True)


@pytest.fixture(scope="module")
def exact_explanation(frozen_result):
    return interpretation.build_latest_interpretation(deepcopy(frozen_result))


def _observation(feature, value=0.):
    return dict(feature=feature, value=value, status="present" if value else "absent",
                observation="Проверяемый факт из ролика.", episodes=[])


def _evidence_row(frozen_result, request_id="independent-1", overrides=None):
    frame = pd.DataFrame(frozen_result["feature_rows"]["n"])
    return dict(source_sha=str(frame.sha.iloc[0]), prompt_sha=evidence_api.PROMPT_SHA,
                request_id=request_id, evidence_version=3,
                values=dict(synopsis="Наблюдаемый сюжет.", main_claim="Предложение сервиса.",
                            observations=[_observation(feature, (overrides or {}).get(feature, 0.))
                                          for feature in evidence_api.DEFS]))


def _validate(row, result):
    frame = pd.DataFrame(result["feature_rows"]["n"])
    return evidence_api.validate_evidence(row, str(frame.sha.iloc[0]),
                                          float(frame.phys__duration.iloc[0]))


def _attach_alignment(parent, repeat, rating=2):
    """Construct explicit action/result observations under the shipped rubric."""
    parent["repeat"] = repeat
    parent["prepared_sha256"] = "a" * 64
    shown = dict(status="shown", representation="direct_visual",
                 description="Герой публикует объявление.",
                 episodes=[dict(start=1., end=2., description="Публикация объявления.")])
    result = dict(status="not_shown", representation="text_or_voice_only",
                  description="Продажа обещана в озвучке, но не показана.", episodes=[])
    if rating == 3:
        result = dict(status="shown", representation="direct_visual",
                      description="Герой передает проданную вещь покупателю.",
                      episodes=[dict(start=3., end=4., description="Передача вещи покупателю.")])
    parent["alignment_evidence"] = dict(
        source_sha256=parent["source_sha"], prepared_sha256="a" * 64, repeat=repeat,
        contract_sha256=alignment_api.CONTRACT_SHA, prompt_sha256=alignment_api.PROMPT_SHA,
        request_id=f"separate-alignment-{repeat}",
        values=dict(main_phrase="Разместите объявление и продайте вещь на Авито.",
                    main_phrase_modality="voice",
                    main_phrase_episodes=[dict(start=1., end=2., description="Главная фраза озвучки.")],
                    claimed_action="Публикация объявления.", claimed_result="Продажа вещи.",
                    action_evidence=shown, result_evidence=result,
                    same_causal_link_in_voice=rating == 3, relation="related", rating=rating,
                    rating_reason="Действие показано; результат " + ("показан." if rating == 3 else "только обещан.")))
    return parent


def _alignment_rows(frozen_result, ratings):
    return [_attach_alignment(_evidence_row(frozen_result, f"general-review-{repeat}"), repeat, rating)
            for repeat, rating in enumerate(ratings, start=1)]


def test_old_alignment_contract_keeps_strict_validation(frozen_result):
    rows = _alignment_rows(frozen_result, [3, 3, 3])
    for parent in rows:
        item = parent['alignment_evidence']
        item['contract_sha256'], item['prompt_sha256'] = next(iter(alignment_api.LEGACY_CONTRACTS))
    assert interpretation.alignment_support(rows, observed=3., duration=30.)['verified']
    rows[0]['alignment_evidence']['values']['rating'] = 2
    assert not interpretation.alignment_support(rows, observed=3., duration=30.)['verified']


def _fast_negative_shap(x, background, predict):
    """Every feature is a stable limit, so a missing gate cannot hide in a tie."""
    contributions = np.full_like(x, -.01, dtype=float)
    bases = predict(x) - contributions.sum(axis=1)
    return contributions, bases


@pytest.mark.parametrize("task", list("nmr"))
def test_design_predictor_exactly_matches_each_frozen_head(frozen_rows, task):
    head = runtime.load_latest_models().heads[task]
    train = _background(task)
    design, background, predict = interpretation.design_predictor(head, frozen_rows[task], train)
    np.testing.assert_allclose(predict(design), head.predict(frozen_rows[task]), atol=1e-12, rtol=0)
    np.testing.assert_allclose(predict(background), head.predict(train), atol=1e-12, rtol=0)
    assert design.shape == (10, len(head.state["columns"]))
    if task == "m":
        assert head.state["spec"]["md_refinement"]
        # This fixture actually exercises the post-log affine calibration.
        assert not (head.state["calibration_intercept"] == 0
                    and head.state["calibration_slope"] == 1)
    if task == "r":
        assert len(head.state["spec"]["members"]) == 2
        members = [type(head)(state, estimator) for state, estimator in head.estimator]
        expected = np.mean([member.predict(frozen_rows[task]) for member in members], axis=0)
        np.testing.assert_allclose(predict(design), expected, atol=1e-12, rtol=0)


@pytest.mark.parametrize("task", list("nmr"))
def test_exact_shap_reconstructs_every_repeat_and_average(frozen_result, exact_explanation, task):
    details = exact_explanation["details"][task]
    values = np.asarray(details["per_repeat_shap"])
    predictions = np.asarray(details["per_repeat_scores"])
    assert values.shape == (10, len(details["feature_order"]))
    np.testing.assert_allclose(details["expected"] + values.sum(axis=1), predictions,
                               atol=1e-9, rtol=0)
    expected_score = frozen_result["scores"][runtime.SCORE_NAMES[task]]
    assert details["expected"] + values.mean(axis=0).sum() == pytest.approx(expected_score, abs=1e-9)
    assert details["actual"] == pytest.approx(expected_score, abs=1e-12)
    assert details["additivity_error"] < 1e-9
    card = next(card for card in exact_explanation["cards"] if card["task"] == task)
    assert card["index"] == pytest.approx(100 * expected_score / details["expected"], abs=1e-9)


def test_exact_shap_on_known_interaction_satisfies_dummy_and_symmetry():
    # Independent analytic case: at x=(1,1,9), f=a*b gives equal halves, while
    # the unused third input receives zero for all possible coalitions.
    query = np.array([[1., 1., 9.], [1., 1., 9.]])
    background = np.array([[0., 0., 0.]])
    values, bases = interpretation.exact_shap(query, background, lambda z: z[:, 0] * z[:, 1])
    np.testing.assert_allclose(values, [[.5, .5, 0.], [.5, .5, 0.]], atol=1e-12, rtol=0)
    np.testing.assert_array_equal(bases, [0., 0.])


def test_evidence_cannot_change_score_or_mutate_scoring_inputs(frozen_result, monkeypatch):
    monkeypatch.setattr(interpretation, "exact_shap", _fast_negative_shap)
    result = deepcopy(frozen_result)
    before = deepcopy(result)
    known = [_evidence_row(result, f"known-{i}") for i in range(3)]
    other = deepcopy(known)
    for row in other:
        row["values"]["synopsis"] = "Совершенно иной текст описания."
        for observation in row["values"]["observations"]:
            observation.update(value=None, status="uncertain", observation="Недостаточно данных.")
    first = interpretation.build_latest_interpretation(result, evidence=known)
    second = interpretation.build_latest_interpretation(result, evidence=other)
    for task in "nmr":
        expected = result["scores"][runtime.SCORE_NAMES[task]]
        assert first["details"][task]["actual"] == pytest.approx(expected, abs=1e-12)
        assert second["details"][task]["actual"] == pytest.approx(expected, abs=1e-12)
        pd.testing.assert_frame_equal(pd.DataFrame(result["feature_rows"][task]),
                                      pd.DataFrame(before["feature_rows"][task]))
    assert result["scores"] == before["scores"]
    assert result["per_repeat"] == before["per_repeat"]
    assert runtime.score_feature_rows(result["feature_rows"])["scores"] == before["scores"]


@pytest.mark.parametrize("field,bad_value", [("source_sha", "another-video"),
                                            ("prompt_sha", "another-contract")])
def test_evidence_rejects_wrong_source_or_prompt(frozen_result, field, bad_value):
    row = _evidence_row(frozen_result)
    row[field] = bad_value
    with pytest.raises(ValueError, match="mismatch"):
        _validate(row, frozen_result)


def test_strict_gate_uses_raw_frame_sha_and_rejects_duplicate_requests(frozen_result, monkeypatch):
    monkeypatch.setattr(interpretation, "exact_shap", _fast_negative_shap)
    result = deepcopy(frozen_result)
    result["source_sha"] = "untrusted-top-level-metadata"
    valid = _evidence_row(result, "same-request")
    wrong_sha = _evidence_row(result, "wrong-source")
    wrong_sha["source_sha"] = result["source_sha"]
    wrong_prompt = _evidence_row(result, "wrong-prompt")
    wrong_prompt["prompt_sha"] = "old-contract"
    missing_request = _evidence_row(result)
    missing_request.pop("request_id")
    response = interpretation.build_latest_interpretation(
        result, evidence=[valid, deepcopy(valid), wrong_sha, wrong_prompt, missing_request])
    assert response["evidence_runs"] == 1
    assert response["evidence_rejected"] == 4
    assert all(not driver["usable"] for detail in response["details"].values()
               for driver in detail["drivers"] if driver["feature"] in evidence_api.DEFS)


@pytest.mark.parametrize("feature", sorted(evidence_api.BINARY))
@pytest.mark.parametrize("value", [True, False])
def test_provider_booleans_are_normalized_only_for_binary_features(frozen_result, feature, value):
    row = _evidence_row(frozen_result, overrides={feature: value})
    checked = _validate(row, frozen_result)
    observation = next(obs for obs in checked["values"]["observations"] if obs["feature"] == feature)
    assert type(observation["value"]) is int
    assert observation["value"] == int(value)


@pytest.mark.parametrize("feature", ["character_close_up_seconds", "fresh__audiovisual_claim_alignment"])
@pytest.mark.parametrize("value", [True, False])
def test_booleans_are_rejected_for_nonbinary_measurements(frozen_result, feature, value):
    row = _evidence_row(frozen_result, overrides={feature: value})
    with pytest.raises(ValueError, match="Nonnumeric"):
        _validate(row, frozen_result)


@pytest.mark.parametrize("value", ["1", "true", "false", "NaN", float("nan"), float("inf")])
def test_strings_and_nonfinite_values_never_become_observations(frozen_result, value):
    row = _evidence_row(frozen_result, overrides={"monologue_to_camera": value})
    with pytest.raises(ValueError, match="Nonnumeric"):
        _validate(row, frozen_result)


@pytest.mark.parametrize("feature,value,reference", [
    ("fresh__audiovisual_claim_alignment", 3., 1.),
    ("panel__offer_novelty_explanation_need", 0., 2.),
    ("brand_first_mention_seconds", 1., 8.),
])
def test_good_observations_penalized_by_model_never_generate_edit_advice(
        frozen_result, monkeypatch, feature, value, reference):
    kind, explanation = interpretation._reading_kind(feature, value, reference, -.02)
    assert kind == "association_only"
    assert explanation
    monkeypatch.setattr(interpretation, "exact_shap", _fast_negative_shap)
    result = deepcopy(frozen_result)
    for rows in result["feature_rows"].values():
        for row in rows:
            if feature in row:
                row[feature] = value
    evidence = [_evidence_row(result, f"independent-{i}", {feature: value}) for i in range(3)]
    if feature == "fresh__audiovisual_claim_alignment":
        evidence = [_attach_alignment(parent, i, rating=3)
                    for i, parent in enumerate(evidence, start=1)]
    response = interpretation.build_latest_interpretation(result, evidence=evidence)
    driver = next(driver for detail in response["details"].values()
                  for driver in detail["drivers"] if driver["feature"] == feature)
    assert driver["evidence"]["verified"]
    assert driver["usable"]
    assert driver["contribution"] < 0
    assert driver["interpretation_kind"] == "association_only"
    assert driver["check"] is None


def test_unverified_or_unknown_features_do_not_become_manager_claims(frozen_result, monkeypatch):
    monkeypatch.setattr(interpretation, "exact_shap", _fast_negative_shap)
    no_evidence = interpretation.build_latest_interpretation(deepcopy(frozen_result))
    for detail in no_evidence["details"].values():
        for driver in detail["drivers"]:
            if driver["feature"] in evidence_api.DEFS:
                assert not driver["usable"]
                assert driver["check"] is None
    forged = _evidence_row(frozen_result)
    forged["values"]["observations"][0]["feature"] = "invented_conversion_potential"
    with pytest.raises(ValueError, match="Incomplete"):
        _validate(forged, frozen_result)
    checked = interpretation.build_latest_interpretation(deepcopy(frozen_result), evidence=[forged])
    assert checked["evidence_runs"] == 0
    assert checked["evidence_rejected"] == 1
    assert all(driver["feature"] != "invented_conversion_potential"
               for detail in checked["details"].values() for driver in detail["drivers"])


def test_celebrity_claim_needs_external_confirmation_even_with_video_agreement(frozen_result):
    rows = [_evidence_row(frozen_result, f"celebrity-{i}", {"is_celeb": 1.}) for i in range(3)]
    support = interpretation._evidence("is_celeb", rows, observed=1., duration=30.)
    assert support["verified"] is False
    assert not support["episodes"]


@pytest.mark.parametrize("status", ["invented", "", None, 1])
def test_evidence_rejects_status_outside_the_contract(frozen_result, status):
    row = _evidence_row(frozen_result)
    row["values"]["observations"][0]["status"] = status
    with pytest.raises(ValueError, match="status"):
        _validate(row, frozen_result)


@pytest.mark.parametrize("feature", sorted(evidence_api.BINARY))
@pytest.mark.parametrize("value", [-1., .5, 2., 3.])
def test_binary_evidence_rejects_values_outside_zero_and_one(frozen_result, feature, value):
    row = _evidence_row(frozen_result, overrides={feature: value})
    with pytest.raises(ValueError, match="binary"):
        _validate(row, frozen_result)


@pytest.mark.parametrize("status,value", [("present", 1.), ("absent", 0.), ("uncertain", None)])
def test_evidence_accepts_defined_statuses_with_valid_values(frozen_result, status, value):
    row = _evidence_row(frozen_result, overrides={"monologue_to_camera": value})
    observation = next(obs for obs in row["values"]["observations"]
                       if obs["feature"] == "monologue_to_camera")
    observation["status"] = status
    assert _validate(row, frozen_result) is row


@pytest.mark.parametrize("dispute", ["conflicting_video_evidence", "unstable_contribution"])
def test_large_disputed_contribution_stays_visible_as_partial_without_changing_score(
        frozen_result, monkeypatch, dispute):
    result = deepcopy(frozen_result)
    before = deepcopy(result)
    feature = "character_close_up_seconds"
    columns = runtime.load_latest_models().heads["n"].state["columns"]
    position = columns.index(feature)
    frame = pd.DataFrame(result["feature_rows"]["n"])
    observed = float(frame[feature].mean())
    duration = float(frame.phys__duration.iloc[0])
    reported = observed
    if dispute == "conflicting_video_evidence":
        # A contradictory but physically possible observation, well outside the
        # accepted time tolerance. The disagreement must be shown to the reader.
        reported = 0. if observed > duration * .1 + 1. else duration

    def disputed_shap(x, background, predict):
        values = np.zeros_like(x, dtype=float)
        values[:, position] = -.3
        if dispute == "unstable_contribution":
            values[-3:, position] = .3
        return values, predict(x) - values.sum(axis=1)

    monkeypatch.setattr(interpretation, "exact_shap", disputed_shap)
    rows = [_evidence_row(result, f"dispute-{i}", {feature: reported}) for i in range(3)]
    actual = interpretation.build_latest_interpretation(result, evidence=rows)
    card = next(card for card in actual["cards"] if card["task"] == "n")
    assert card["explanation_status"] == "partial"
    assert card["verified_contribution_coverage"] < .75
    disputed = next(driver for driver in card["unresolved_drivers"] if driver["feature"] == feature)
    assert disputed["label"] in card["unresolved"]
    assert abs(disputed["index_points"]) >= 5
    assert disputed["support_reason"]
    assert disputed["check"] is None
    assert not disputed["usable"]
    if dispute == "conflicting_video_evidence":
        assert disputed["evidence"]["verified"] is False
    else:
        assert disputed["evidence"]["verified"] is True
        assert disputed["stable_fraction"] == pytest.approx(.7)
    assert feature not in {driver["feature"] for driver in card["strengths"] + card["limitations"]}
    for task in "nmr":
        expected = before["scores"][runtime.SCORE_NAMES[task]]
        assert actual["details"][task]["actual"] == pytest.approx(expected, abs=1e-12)
        pd.testing.assert_frame_equal(pd.DataFrame(result["feature_rows"][task]),
                                      pd.DataFrame(before["feature_rows"][task]))
    assert result["scores"] == before["scores"]
    assert runtime.score_feature_rows(result["feature_rows"])["scores"] == before["scores"]


@pytest.mark.parametrize("feature", ["has_callback_to_opening", "jingle_present",
                                     "problem_solution_arc_present", "scene_pace_high"])
@pytest.mark.parametrize("positive_count,expected", [(7, 1.), (6, 0.)])
def test_recall_semantic_measurement_uses_frozen_two_thirds_consensus_without_mutation(
        frozen_rows, feature, positive_count, expected):
    frame = frozen_rows["r"].copy(deep=True)
    frame[feature] = [1.] * positive_count + [0.] * (10 - positive_count)
    before = frame.copy(deep=True)
    head = runtime.load_latest_models().heads["r"]
    prediction_before = head.predict(frame)
    used_value, measurement_mean = interpretation.semantic_measurement(head, frame, feature)
    assert used_value == expected
    assert measurement_mean == pytest.approx(positive_count / 10)
    pd.testing.assert_frame_equal(frame, before)
    np.testing.assert_array_equal(head.predict(frame), prediction_before)


def test_semantic_measurement_keeps_nonconsensus_measurement_in_original_units(frozen_rows):
    head = runtime.load_latest_models().heads["r"]
    frame = frozen_rows["r"].copy(deep=True)
    frame["brand_first_mention_seconds"] = np.arange(10, dtype=float)
    before = frame.copy(deep=True)
    value, original = interpretation.semantic_measurement(head, frame, "brand_first_mention_seconds")
    assert value == original == 4.5
    pd.testing.assert_frame_equal(frame, before)


def test_structured_alignment_rating_two_does_not_confirm_original_three(frozen_result):
    rows = _alignment_rows(frozen_result, [2, 2, 2])
    support = interpretation.alignment_support(rows, observed=3., duration=30.)
    assert support["verified"] is False
    assert support["agreement"] == 0
    assert support["episodes"] == []
    assert support["source"] == "structured_alignment_review"


@pytest.mark.parametrize("observed,ratings", [(2., [2, 2, 3]), (3., [3, 3, 2])])
def test_structured_alignment_two_out_of_three_is_not_confirmed(
        frozen_result, observed, ratings):
    rows = _alignment_rows(frozen_result, ratings)
    before = deepcopy(rows)
    support = interpretation.alignment_support(rows, observed=observed, duration=30.)
    assert support["verified"] is False
    assert support["agreement"] == pytest.approx(2 / 3)
    assert support["episodes"] == []
    assert support["source"] == "structured_alignment_review"
    assert rows == before


@pytest.mark.parametrize("rating", [2, 3])
def test_structured_alignment_three_distinct_matching_reads_are_confirmed(frozen_result, rating):
    rows = _alignment_rows(frozen_result, [rating] * 3)
    before = deepcopy(rows)
    support = interpretation.alignment_support(rows, observed=float(rating), duration=30.)
    assert support["verified"] is True
    assert support["agreement"] == 1.
    assert support["episodes"]
    assert support["source"] == "structured_alignment_review"
    assert rows == before


@pytest.mark.parametrize("field", ["source_sha256", "prepared_sha256", "prompt_sha256", "contract_sha256"])
def test_alignment_support_does_not_accept_an_invalid_source_or_contract_hash(frozen_result, field):
    rows = _alignment_rows(frozen_result, [2, 2])
    rows[1]["alignment_evidence"][field] = "different-hash"
    support = interpretation.alignment_support(rows, observed=2., duration=30.)
    assert support["verified"] is False
    assert support["agreement"] == pytest.approx(1 / 3)


@pytest.mark.parametrize("bad_schema", ["missing_required", "unexpected_field", "string_rating",
                                       "rating_without_visual_result"])
def test_alignment_support_rejects_invalid_schema_and_unsupported_rating(frozen_result, bad_schema):
    rows = _alignment_rows(frozen_result, [2, 2])
    values = rows[1]["alignment_evidence"]["values"]
    if bad_schema == "missing_required":
        values.pop("rating_reason")
    elif bad_schema == "unexpected_field":
        values["unrequested_quality_score"] = 100
    elif bad_schema == "string_rating":
        values["rating"] = "2"
    else:
        values["rating"] = 3
    support = interpretation.alignment_support(rows, observed=2., duration=30.)
    assert support["verified"] is False
    assert support["agreement"] == pytest.approx(1 / 3)


@pytest.mark.parametrize("bad_id", ["duplicate_alignment", "same_as_general_review", "missing", "blank"])
def test_alignment_support_rejects_reused_or_missing_request_identity(frozen_result, bad_id):
    rows = _alignment_rows(frozen_result, [2, 2])
    if bad_id == "duplicate_alignment":
        rows[1]["alignment_evidence"]["request_id"] = rows[0]["alignment_evidence"]["request_id"]
    elif bad_id == "same_as_general_review":
        rows[1]["alignment_evidence"]["request_id"] = rows[1]["request_id"]
    elif bad_id == "missing":
        rows[1]["alignment_evidence"].pop("request_id")
    else:
        rows[1]["alignment_evidence"]["request_id"] = "   "
    support = interpretation.alignment_support(rows, observed=2., duration=30.)
    assert support["verified"] is False
    assert support["agreement"] == pytest.approx(1 / 3)


def test_general_video_reviews_cannot_replace_structured_alignment_evidence(frozen_result):
    rows = [_evidence_row(frozen_result, f"unstructured-{i}",
                          {"fresh__audiovisual_claim_alignment": 3.}) for i in range(3)]
    support = interpretation.alignment_support(rows, observed=3., duration=30.)
    assert support["verified"] is False
    assert support["agreement"] == 0


@pytest.mark.parametrize("bad_parent", ["missing_prepared_hash", "different_prepared_hash"])
def test_alignment_requires_an_independent_prepared_media_hash(frozen_result, bad_parent):
    rows = _alignment_rows(frozen_result, [2, 2])
    if bad_parent == "missing_prepared_hash":
        rows[1].pop("prepared_sha256")
    else:
        rows[1]["prepared_sha256"] = "b" * 64
    support = interpretation.alignment_support(rows, observed=2., duration=30.)
    assert support["verified"] is False
    assert support["agreement"] == pytest.approx(1 / 3)


@pytest.mark.parametrize("values,verified", [([1., 1., 0.], False), ([1., 1., 1.], True)])
def test_regular_video_observation_requires_all_three_matching_reads(frozen_result, values, verified):
    rows = [_evidence_row(frozen_result, f"monologue-{i}", {"monologue_to_camera": value})
            for i, value in enumerate(values)]
    support = interpretation._evidence("monologue_to_camera", rows, observed=1., duration=30.)
    assert support["verified"] is verified
    assert support["agreement"] == pytest.approx(sum(values) / 3)


@pytest.mark.parametrize("text", [
    "В кадре Николай Фоменко говорит о сервисе.",
    "Дмитрий Журавлёв обращается к собеседнику.",
    "Равшана Куркова показывает объявление.",
    "Герой похож на Фоменко.",
    "Мария Иванова передает вещь покупателю.",
])
def test_name_gate_detects_unsupported_person_mentions(text):
    assert interpretation.has_person_name(text)


@pytest.mark.parametrize("text", [
    "Герой разговаривает по видеосвязи, обращения к зрителю нет.",
    "В ролике показана Авито Работа и поиск вакансии.",
    "На экране Авито Доставка и оформление заказа.",
    "Главная героиня публикует объявление.",
])
def test_name_gate_keeps_anonymous_observations_and_product_names(text):
    assert not interpretation.has_person_name(text)


@pytest.mark.parametrize("location", ["observation", "episode"])
def test_unconfirmed_name_blocks_regular_observation_even_with_three_matching_values(
        frozen_result, location):
    rows = [_evidence_row(frozen_result, f"named-{i}", {"monologue_to_camera": 1.})
            for i in range(3)]
    first = next(obs for obs in rows[0]["values"]["observations"]
                 if obs["feature"] == "monologue_to_camera")
    if location == "observation":
        first["observation"] = "Николай Фоменко смотрит в камеру."
    else:
        first["episodes"] = [dict(start=1., end=2., description="Николай Фоменко смотрит в камеру.")]
    support = interpretation._evidence("monologue_to_camera", rows, observed=1., duration=30.)
    assert support["verified"] is False
    assert support["episodes"] == []
    assert "Фоменко" not in support["observation"]


@pytest.mark.parametrize("location", ["rating_reason", "episode"])
def test_unconfirmed_name_blocks_structured_alignment_even_with_three_matching_ratings(
        frozen_result, location):
    rows = _alignment_rows(frozen_result, [3, 3, 3])
    values = rows[0]["alignment_evidence"]["values"]
    if location == "rating_reason":
        values["rating_reason"] = "Николай Фоменко показывает действие и результат."
    else:
        values["action_evidence"]["episodes"][0]["description"] = "Николай Фоменко публикует объявление."
    support = interpretation.alignment_support(rows, observed=3., duration=30.)
    assert support["verified"] is False
    assert support["episodes"] == []
    assert "Фоменко" not in support["observation"]


def test_alignment_disagreement_does_not_repeat_an_unconfirmed_person_name(frozen_result):
    rows = _alignment_rows(frozen_result, [2, 2, 2])
    rows[0]["alignment_evidence"]["values"]["rating_reason"] = "Николай Фоменко показал действие, но не результат."
    support = interpretation.alignment_support(rows, observed=3., duration=30.)
    assert support["verified"] is False
    assert "Фоменко" not in support["observation"]
    assert support["episodes"] == []


def test_alignment_name_gate_also_covers_fallback_main_phrase_episodes(frozen_result):
    rows = _alignment_rows(frozen_result, [2, 2, 2])
    for row in rows:
        values = row["alignment_evidence"]["values"]
        values["action_evidence"] = dict(status="not_shown", representation="text_or_voice_only",
                                          description="Действие только названо в озвучке.", episodes=[])
        values["rating"] = 1
        values["rating_reason"] = "Изображение задает тему; действие и результат не показаны."
    rows[0]["alignment_evidence"]["values"]["main_phrase_episodes"][0]["description"] = "Николай Фоменко произносит главную фразу."
    support = interpretation.alignment_support(rows, observed=1., duration=30.)
    assert support["verified"] is False
    assert "Фоменко" not in support["observation"]
    assert support["episodes"] == []


@pytest.mark.parametrize("reason", ["two_of_three", "unconfirmed_name", "human_count"])
def test_conservative_evidence_gates_keep_numeric_driver_and_scores_unchanged(
        frozen_result, monkeypatch, reason):
    rows = deepcopy(frozen_result["feature_rows"])
    feature, task = ("human_characters_count", "r") if reason == "human_count" else ("monologue_to_camera", "n")
    if reason != "human_count":
        for frame in rows.values():
            for row in frame:
                if feature in row:
                    row[feature] = 1.
    result = runtime.score_feature_rows(rows)
    before = deepcopy(result)
    value = float(pd.DataFrame(rows[task])[feature].mean())
    evidence = [_evidence_row(result, f"gate-{i}", {feature: value}) for i in range(3)]
    observations = [next(obs for obs in row["values"]["observations"] if obs["feature"] == feature)
                    for row in evidence]
    if reason == "two_of_three":
        observations[2].update(value=0., status="absent",
                               observation="Герой говорит по видеосвязи; обращения к зрителю нет.")
    elif reason == "unconfirmed_name":
        observations[0]["observation"] = "Николай Фоменко обращается к зрителю."
    else:
        for observation in observations:
            observation["observation"] = "В кадре несколько персонажей; состав требует покадровой проверки."

    position = runtime.load_latest_models().heads[task].state["columns"].index(feature)
    def large_contribution(x, background, predict):
        contribution = np.zeros_like(x, dtype=float)
        contribution[:, position] = -.3
        return contribution, predict(x) - contribution.sum(axis=1)

    monkeypatch.setattr(interpretation, "exact_shap", large_contribution)
    actual = interpretation.build_latest_interpretation(result, evidence=evidence)
    card = next(card for card in actual["cards"] if card["task"] == task)
    driver = next(driver for driver in card["unresolved_drivers"] if driver["feature"] == feature)
    assert card["explanation_status"] == "partial"
    assert driver["contribution"] == pytest.approx(-.3)
    assert driver["value"] == pytest.approx(value)
    assert not driver["evidence"]["verified"]
    assert not driver["usable"]
    assert driver["check"] is None
    assert driver["label"] in card["unresolved"]
    for component in "nmr":
        assert actual["details"][component]["actual"] == pytest.approx(
            before["scores"][runtime.SCORE_NAMES[component]], abs=1e-12)
        pd.testing.assert_frame_equal(pd.DataFrame(result["feature_rows"][component]),
                                      pd.DataFrame(before["feature_rows"][component]))
    assert result["scores"] == before["scores"]
    assert runtime.score_feature_rows(result["feature_rows"])["scores"] == before["scores"]


@pytest.mark.parametrize("text,expected", [
    ("На Авито", False),
    ("Логотип Авито", False),
    ("С Авито Работой", False),
    ("Николай Фоменко", True),
    ("MACAN", True),
])
def test_name_gate_distinguishes_avito_phrases_from_known_names(text, expected):
    assert interpretation.has_person_name(text) is expected


def test_public_result_redacts_names_without_mutating_original_scores_or_inputs(
        frozen_result, exact_explanation):
    original = deepcopy(frozen_result)
    row = _alignment_rows(original, [3])[0]
    row["values"]["synopsis"] = "Николай Фоменко публикует объявление."
    row["values"]["main_claim"] = "Дмитрий Журавлёв предлагает продать вещь."
    observations = {obs["feature"]: obs for obs in row["values"]["observations"]}
    observations["monologue_to_camera"].update(
        status="present", value=1., observation="Николай Фоменко обращается к зрителю.")
    observations["pack_shot_duration_seconds"]["episodes"] = [
        dict(start=1., end=2., description="Равшана Куркова рядом с логотипом.")]
    observations["main_character"]["observation"] = "Главный герой показан в начале ролика."
    row["alignment_evidence"]["values"]["rating_reason"] = "MACAN показывает действие и результат."
    original["independent_evidence"] = [row]
    original["evidence"] = [deepcopy(row)]
    original.setdefault("metadata", {})["evidence"] = [deepcopy(row)]
    before = deepcopy(original)
    before_interpretation = deepcopy(exact_explanation)

    published = interpretation.public_result(original, exact_explanation)

    assert "evidence" not in published
    assert "evidence" not in published["metadata"]
    public_row = published["independent_evidence"][0]
    assert "alignment_evidence" not in public_row
    for name in ("Фоменко", "Журавлёв", "Куркова", "MACAN"):
        assert name not in repr(public_row)
    public_observations = {obs["feature"]: obs for obs in public_row["values"]["observations"]}
    for feature in ("monologue_to_camera", "pack_shot_duration_seconds", "is_celeb"):
        assert public_observations[feature]["status"] == "uncertain"
        assert public_observations[feature]["value"] is None
        assert public_observations[feature]["episodes"] == []
    assert public_observations["main_character"] == observations["main_character"]
    assert {"synopsis", "main_claim", "monologue_to_camera", "pack_shot_duration_seconds",
            "alignment_evidence"}.issubset(public_row["public_redactions"])
    assert original["independent_evidence"] == before["independent_evidence"]
    assert original["evidence"] == before["evidence"]
    assert original["metadata"] == before["metadata"]
    assert published["scores"] == original["scores"] == before["scores"]
    assert published["per_repeat"] == original["per_repeat"] == before["per_repeat"]
    assert published["interpretation"] == exact_explanation == before_interpretation
    assert published["interpretation"] is not exact_explanation
    for task in "nmr":
        expected = pd.DataFrame(before["feature_rows"][task])
        pd.testing.assert_frame_equal(pd.DataFrame(original["feature_rows"][task]), expected)
        pd.testing.assert_frame_equal(pd.DataFrame(published["feature_rows"][task]), expected)
    assert runtime.score_feature_rows(published["feature_rows"])["scores"] == before["scores"]


def test_confirmed_fact_with_different_scoring_value_stays_separate(frozen_result):
    rows = [_evidence_row(frozen_result, f'fact-{i}', overrides={
        'product_demo_seconds': 0
    }) for i in range(3)]
    # Use the same validated observations; this helper must not substitute them into inference.
    duration = 30
    fact = interpretation.factual_observation('product_demo_seconds', rows, duration)
    assert fact['value'] == 0 and fact['evidence']['verified']
    assert 'Без демонстрации' in fact['label']


def test_legacy_evidence_contract_is_readable_but_unknown_contract_is_not(frozen_result):
    row = _evidence_row(frozen_result)
    frame = pd.DataFrame(frozen_result['feature_rows']['n'])
    row['prompt_sha'] = evidence_api.LEGACY_PROMPT_SHA
    evidence_api.validate_evidence(row, str(frame.sha.iloc[0]), float(frame['phys__duration'].iloc[0]))
    row['prompt_sha'] = 'unknown'
    with pytest.raises(ValueError):
        evidence_api.validate_evidence(row, str(frame.sha.iloc[0]), float(frame['phys__duration'].iloc[0]))


@pytest.mark.parametrize('phrase', ['Дед Мороз вручает подарок.', 'Героиня обращается к Деду Морозу.',
                                    'Встреча с Дедом Морозом показана в финале.'])
def test_fictional_holiday_role_is_not_a_real_person_identity(phrase):
    assert not interpretation.has_person_name(phrase)
    assert interpretation.has_person_name(phrase + ' Николай Фоменко в кадре.')


def test_cached_review_keeps_existing_v3_observations_without_live_calls(frozen_result, tmp_path, monkeypatch):
    source = tmp_path / 'video.mp4'
    source.write_bytes(b'fixture')
    sha = 'a' * 64
    monkeypatch.setattr(evidence_api, 'file_sha256', lambda path: sha)
    monkeypatch.setattr(evidence_api.md, 'prepare_video', lambda path, dest: source)
    monkeypatch.setattr(evidence_api.md, 'video_duration', lambda path: 30.)
    root = tmp_path / 'independent-video-evidence-v3' / sha
    root.mkdir(parents=True)
    for repeat in range(1, 4):
        row = _evidence_row(frozen_result, f'legacy-{repeat}')
        row.update(source_sha=sha, prompt_sha=evidence_api.LEGACY_PROMPT_SHA, repeat=repeat)
        (root / f'repeat_{repeat:02}.json').write_text(json.dumps(row))
    rows = evidence_api.collect_independent_evidence(source_video=source, output_root=tmp_path, allow_live=False)
    assert len(rows) == 3
    assert all(row['prompt_sha'] == evidence_api.LEGACY_PROMPT_SHA for row in rows)
