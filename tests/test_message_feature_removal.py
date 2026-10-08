"""Eight-input scoring, unchanged N/R, cache isolation and explanation parity."""
from copy import deepcopy
import json

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

from aipm3 import latest_runtime as runtime
from aipm3.latest_interpretation import build_latest_interpretation
from aipm3.latest_profile_ui import profile_rows
from aipm3.neuromatics_models import NeuromaticsModels

FEATURE = "numeric_offer_on_screen"


@pytest.fixture(scope="module")
def frames():
    rows = {}
    for task in "nmr":
        fit = pd.read_csv(runtime.BUNDLE_DIR / "data" / f"fit_{task}.csv", float_precision="round_trip")
        frame = pd.DataFrame([fit.iloc[0].to_dict()] * 3)
        frame["record"] = frame["family"] = "m8-regression"
        frame["sha"] = "b" * 64
        frame["repeat"] = [1, 2, 3]
        frame["brand"], frame["vertical"] = "Avito", "Goods"
        rows[task] = frame.drop(columns=[FEATURE], errors="ignore")
    return rows


def test_new_artifact_is_exact_full_refit_and_original_bundles_are_preserved():
    module = runtime.model_module()
    old = module.AIPM3(runtime.BUNDLE_DIR / "models")
    current = runtime.load_latest_models()
    spec = deepcopy(old.heads["m"].state["spec"])
    spec["fixed"].remove(FEATURE)
    train = pd.read_csv(runtime.BUNDLE_DIR / "data/fit_m.csv", float_precision="round_trip")
    expected = module.Head.fit(train.drop(columns=[FEATURE]), spec)
    np.testing.assert_array_equal(current.heads["m"].predict(train), expected.predict(train))
    assert current.heads["m"].state == expected.state
    assert [len(current.heads[t].state["columns"]) for t in "nmr"] == [9, 8, 7]
    for task in "nr":
        frame = pd.read_csv(runtime.BUNDLE_DIR / "data" / f"fit_{task}.csv", float_precision="round_trip")
        np.testing.assert_array_equal(current.heads[task].predict(frame), old.heads[task].predict(frame))
    runtime.artifact_hashes()
    runtime.message_artifact_hashes()


@pytest.mark.parametrize("kind", ["finished", "neuromatics"])
def test_absent_or_arbitrary_numeric_input_never_changes_scores_or_explanations(frames, kind):
    expected = runtime.score_feature_rows(frames, material_kind=kind, repeat_count=3)
    changed = {task: frame.assign(**{FEATURE: [0., 1., 999.]}) for task, frame in frames.items()}
    actual = runtime.score_feature_rows(changed, material_kind=kind, repeat_count=3)
    assert actual["scores"] == expected["scores"]
    assert actual["per_repeat"] == expected["per_repeat"]
    assert FEATURE not in actual["model_inputs"]["m"]
    assert runtime.validate_cached_result(expected)["scores"] == expected["scores"]
    interpretation = build_latest_interpretation(expected)
    assert len(interpretation["details"]["m"]["feature_order"]) == 8
    assert FEATURE not in interpretation["details"]["m"]["feature_order"]
    assert FEATURE not in {row["feature"] for row in profile_rows(interpretation)["m"]}
    assert interpretation["details"]["m"]["additivity_error"] < 1e-9
    module = runtime.model_module()
    old = module.AIPM3(runtime.BUNDLE_DIR / "models")
    if kind == "neuromatics":
        old.heads["m"] = runtime.load_latest_models().heads["m"]
        old = NeuromaticsModels(old, module.coefficient)
    for task in "nr":
        np.testing.assert_array_equal(runtime.load_models(kind).heads[task].predict(frames[task]),
                                      old.heads[task].predict(frames[task]))


def test_stored_loo_matches_previously_checked_ablation():
    frame = pd.read_csv(runtime.MESSAGE_BUNDLE_DIR / "human_loo_m.csv", float_precision="round_trip")
    assert len(frame) == 58 and frame.family.nunique() == 56
    assert spearmanr(frame.y, frame.prediction).statistic == pytest.approx(.6455372214787949, abs=1e-12)
    assert spearmanr(frame.y, frame.previous).statistic == pytest.approx(.6215365773775191, abs=1e-12)


def test_display_references_follow_new_message_head_without_changing_nr_calibration():
    current = json.loads((runtime.MESSAGE_BUNDLE_DIR / "neuromatics_display_reference.json").read_text())
    original = json.loads((runtime.BUNDLE_DIR.parents[1] / "neuromatics_display_reference.json").read_text())
    assert current["scoring_version"] == runtime.NEUROMATICS_SCORING_VERSION
    for task in "nr":
        assert current["references"][task] == original["references"][task]
    train = pd.read_csv(runtime.BUNDLE_DIR / "data/fit_m.csv", float_precision="round_trip")
    bg = train.sort_values(["family", "record", "repeat"]).groupby("family", sort=True).nth(0)
    reference = current["references"]["m"]
    pred = runtime.load_latest_models().heads["m"].predict(bg)
    assert reference["finished_reference"]["mean"] == pytest.approx(pred.mean(), abs=1e-12)
    np.testing.assert_allclose(reference["finished_reference"]["cuts"], np.quantile(pred, [1 / 3, 2 / 3]))


@pytest.mark.parametrize("kind,old_version", [
    ("finished", "three-heads-effect15-20261001-979fdba8b527-f0aa4e015216-6d75e5e23d76"),
    ("neuromatics", "neuromatics-nclip-ridge100-20261002-c9e7282c127d"),
])
def test_old_result_is_not_silently_relabelled_as_new_model(frames, kind, old_version):
    result = runtime.score_feature_rows(frames, material_kind=kind, repeat_count=3)
    result["scoring_version"] = old_version
    with pytest.raises(ValueError, match="версией"):
        runtime.validate_cached_result(result)
