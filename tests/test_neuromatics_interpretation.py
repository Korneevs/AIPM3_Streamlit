"""Exact selected-model attributions and raw-versus-adapted evidence handling.

These offline tests use the shipped historical schemas, without private caches,
provider requests or fitted coefficients created during testing.
"""
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from aipm3 import latest_interpretation as interpretation
from aipm3 import latest_runtime as runtime


@pytest.fixture(scope="module")
def rows():
    result = {}
    for task in "nmr":
        history = pd.read_csv(runtime.BUNDLE_DIR / "data" / f"fit_{task}.csv",
                              float_precision="round_trip")
        frame = pd.DataFrame([history.iloc[0].to_dict() for _ in range(10)])
        frame["record"] = "interpretation-neuromatics"
        frame["family"] = "unseen-interpretation-family"
        frame["brand"] = "Avito"
        frame["vertical"] = "Goods"
        frame["sha"] = "d" * 64
        frame["repeat"] = np.arange(1, 11)
        frame["phys__duration"] = 30.
        frame["total_video_duration_sec"] = 30.
        result[task] = frame
    # Deliberately distinguish raw-file facts, adaptation and consensus.
    result["n"]["phys__audio_dynamic_range_db"] = 50.
    result["n"]["phys__motion_mean"] = 0.
    result["n"]["state_transformation_present"] = 1.
    result["n"]["state_transformation"] = 0.
    result["r"]["jingle_present"] = [1.] * 6 + [0.] * 4
    return result


def _background(task):
    frame = pd.read_csv(runtime.BUNDLE_DIR / "data" / f"fit_{task}.csv",
                        float_precision="round_trip")
    order = [column for column in ("family", "record", "repeat") if column in frame]
    return frame.sort_values(order).groupby("family", sort=True).nth(0).reset_index(drop=True)


@pytest.fixture(scope="module")
def neuro_result(rows):
    return runtime.score_feature_rows(rows, material_kind="neuromatics")


@pytest.fixture(scope="module")
def neuro_explanation(neuro_result):
    return interpretation.build_latest_interpretation(neuro_result)


@pytest.mark.parametrize("task", list("nmr"))
def test_selected_head_design_and_exact_shap_match_every_prediction(rows, neuro_result,
                                                                   neuro_explanation, task):
    head = runtime.load_models("neuromatics").heads[task]
    design, background, predict = interpretation.design_predictor(head, rows[task], _background(task))
    np.testing.assert_allclose(predict(design), head.predict(rows[task]), atol=1e-12, rtol=0)
    np.testing.assert_allclose(predict(background), head.predict(_background(task)), atol=1e-12, rtol=0)
    details = neuro_explanation["details"][task]
    np.testing.assert_allclose(details["expected"] + np.asarray(details["per_repeat_shap"]).sum(axis=1),
                               details["per_repeat_scores"], atol=1e-9, rtol=0)
    assert details["actual"] == pytest.approx(neuro_result["scores"][runtime.SCORE_NAMES[task]], abs=1e-12)
    assert details["additivity_error"] < 1e-9
    assert len(details["feature_order"]) == (7 if task == "r" else 9)


def test_default_finished_interpretation_preserves_frozen_model(rows):
    result = runtime.score_feature_rows(rows)
    default = interpretation.build_latest_interpretation(result)
    frozen = interpretation.build_latest_interpretation(
        result, model=runtime.load_latest_models(), bundle_dir=runtime.BUNDLE_DIR)
    assert runtime.clean_json(default) == runtime.clean_json(frozen)
    assert default["overall"]["title"] == "AIPM 3.0"
    assert default["material_kind"] == "finished"
    for task in "nmr":
        np.testing.assert_allclose(default["details"][task]["per_repeat_scores"],
                                   runtime.load_latest_models().heads[task].predict(rows[task]),
                                   atol=1e-12, rtol=0)


def test_neuromatics_explanation_selects_adapted_heads_and_leaves_message_delivery(rows,
                                                                               neuro_explanation):
    frozen = runtime.load_latest_models()
    assert neuro_explanation["material_kind"] == "neuromatics"
    assert neuro_explanation["overall"]["title"] == "AIPM3.0 (для нейроматиков)"
    np.testing.assert_allclose(neuro_explanation["details"]["m"]["per_repeat_scores"],
                               frozen.heads["m"].predict(rows["m"]), atol=1e-12, rtol=0)
    assert not np.allclose(neuro_explanation["details"]["r"]["per_repeat_scores"],
                           frozen.heads["r"].predict(rows["r"]))


def test_display_transfer_preserves_scores_and_uses_paired_prototype_reference(neuro_result, neuro_explanation):
    from aipm3.display_calibration import neuromatics_reference
    ref = neuromatics_reference(runtime.NEUROMATICS_SCORING_VERSION)
    assert ref['pair_count'] == 12
    for c in neuro_explanation['cards']:
        assert c['score'] == pytest.approx(neuro_result['scores'][runtime.SCORE_NAMES[c['task']]], abs=1e-12)
        assert c['index'] == pytest.approx(100*c['score']/ref['references'][c['task']]['mean'])
    assert neuro_explanation['overall']['index'] == pytest.approx(100*neuro_result['scores']['Q']/ref['references']['Q']['mean'])


def test_incomplete_voice_withholds_norms_without_imputing_or_changing_scores(neuro_result):
    from aipm3.manager_report import report_cards
    before = deepcopy(neuro_result)
    result = deepcopy(neuro_result)
    result['source_sha'] = 'd'*64
    result['audio_review'] = dict(status='partial', source_sha='d'*64)
    explained = interpretation.build_latest_interpretation(result)
    assert not explained['overall']['assessment_available']
    assert all(not c['assessment_available'] for c in explained['cards'])
    assert neuro_result == before
    assert result['scores'] == neuro_result['scores']
    assert explained['audio_status'] == 'partial'
    assert all(item['feature'] not in {'jingle_present','brand_first_mention_seconds','fresh__audiovisual_claim_alignment'}
               for c in report_cards(explained) for g in ['strengths','limitations'] for item in c[g])


def test_physical_evidence_describes_original_file_not_clipped_input(neuro_explanation):
    drivers = {driver["feature"]: driver for driver in neuro_explanation["details"]["n"]["drivers"]}
    audio = drivers["phys__audio_dynamic_range_db"]
    assert audio["original_measurement_mean"] == 50.
    assert audio["value"] < 50.
    assert "50.0 дБ" in audio["evidence"]["observation"]
    assert audio["evidence"]["source"] == "physical_neuromatics_adjusted"
    for feature in ("phys__audio_dynamic_range_db", "phys__motion_mean"):
        assert drivers[feature]["interpretation_kind"] == "association_only"
        assert drivers[feature]["check"] is None


def test_recall_evidence_uses_ten_repeat_consensus(neuro_explanation):
    driver = next(driver for driver in neuro_explanation["details"]["r"]["drivers"]
                  if driver["feature"] == "jingle_present")
    assert driver["original_measurement_mean"] == pytest.approx(.6)
    assert driver["value"] == 0.
    assert driver["label"] == "Без фирменной мелодии"
    assert not driver["usable"]


def test_legacy_state_input_is_the_attributed_meaning(rows):
    head = runtime.load_models("neuromatics").heads["n"]
    frame = rows["n"].copy()
    train = _background("n")
    x, background, predict = interpretation.design_predictor(head, frame, train)
    position = head.state["columns"].index("state_transformation_present")
    # The old broad extraction cannot influence either scores or SHAP designs.
    frame["state_transformation_present"] = 0.
    x_other, _, _ = interpretation.design_predictor(head, frame, train)
    np.testing.assert_array_equal(x, x_other)
    phi, base = interpretation.exact_shap(x[:1], background, predict)
    phi_other, base_other = interpretation.exact_shap(x_other[:1], background, predict)
    np.testing.assert_array_equal(phi, phi_other)
    np.testing.assert_array_equal(base, base_other)
    # A changed stricter reading changes only its existing, named model input.
    frame["state_transformation"] = 1.
    x_legacy, _, _ = interpretation.design_predictor(head, frame, train)
    assert not np.array_equal(x[:, position], x_legacy[:, position])
    np.testing.assert_array_equal(np.delete(x, position, axis=1),
                                   np.delete(x_legacy, position, axis=1))


def test_state_evidence_must_agree_with_selected_legacy_reading(neuro_result, monkeypatch):
    result = deepcopy(neuro_result)
    feature = "state_transformation_present"
    position = runtime.load_models("neuromatics").heads["n"].state["columns"].index(feature)

    def force_driver(x, background, predict):
        values = np.zeros_like(x)
        if x.shape[1] == 9:
            values[:, position] = -.02
        return values, predict(x) - values.sum(axis=1)

    monkeypatch.setattr(interpretation, "exact_shap", force_driver)
    def evidence(value):
        return [dict(values=dict(observations=[dict(feature=feature, value=value,
                     status="present" if value else "absent", episodes=[],
                     observation="Исходное и итоговое состояние показаны." if value
                                 else "Разница между исходной ситуацией и результатом не показана.")]))
                for _ in range(3)]

    matching = interpretation.build_latest_interpretation(result, evidence=evidence(0.), strict_evidence=False)
    conflicting = interpretation.build_latest_interpretation(result, evidence=evidence(1.), strict_evidence=False)
    get_driver = lambda output: next(driver for driver in output["details"]["n"]["drivers"]
                                     if driver["feature"] == feature)
    assert get_driver(matching)["value"] == 0.
    assert get_driver(matching)["evidence"]["verified"]
    assert not get_driver(conflicting)["evidence"]["verified"]
    assert not get_driver(conflicting)["usable"]
    assert get_driver(conflicting)["check"] is None
    assert matching["details"]["n"]["actual"] == conflicting["details"]["n"]["actual"]
