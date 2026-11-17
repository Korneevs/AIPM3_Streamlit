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
        fit = pd.read_csv(runtime.RELEASE_BUNDLE_DIR / "data" / f"fit_{task}.csv", float_precision="round_trip")
        frame = pd.DataFrame([fit.iloc[0].to_dict()] * 3)
        frame["record"] = frame["family"] = "m8-regression"
        frame["sha"] = "b" * 64
        frame["repeat"] = [1, 2, 3]
        frame["brand"], frame["vertical"] = "Avito", "Goods"
        rows[task] = frame.drop(columns=[FEATURE], errors="ignore")
    return rows


def test_release_preserves_original_artifacts_and_excludes_removed_feature():
    runtime.artifact_hashes()
    runtime.message_artifact_hashes()
    runtime.artifact_hashes_for()
    current = runtime.load_latest_models()
    assert [len(current.heads[t].state["columns"]) for t in "nmr"] == [9, 8, 7]
    assert FEATURE not in current.heads["m"].state["columns"]
    assert "panel__content_precision" in current.heads["m"].state["columns"]


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


def test_stored_loo_matches_previously_checked_ablation():
    frame = pd.read_csv(runtime.MESSAGE_BUNDLE_DIR / "human_loo_m.csv", float_precision="round_trip")
    assert len(frame) == 58 and frame.family.nunique() == 56
    assert spearmanr(frame.y, frame.prediction).statistic == pytest.approx(.6455372214787949, abs=1e-12)
    assert spearmanr(frame.y, frame.previous).statistic == pytest.approx(.6215365773775191, abs=1e-12)


def test_display_references_use_the_selected_release_in_both_modes():
    from aipm3.display_calibration import neuromatics_reference
    active = neuromatics_reference(runtime.NEUROMATICS_SCORING_VERSION)
    for task in "nmr":
        assert active["references"][task] == runtime.display_reference()[task]
    assert active["source"] == "selected_family_loo"


@pytest.mark.parametrize("kind,old_version", [
    ("finished", "three-heads-effect15-20261001-979fdba8b527-f0aa4e015216-6d75e5e23d76"),
    ("neuromatics", "neuromatics-nclip-ridge100-20261002-c9e7282c127d"),
])
def test_old_result_is_not_silently_relabelled_as_new_model(frames, kind, old_version):
    result = runtime.score_feature_rows(frames, material_kind=kind, repeat_count=3)
    result["scoring_version"] = old_version
    with pytest.raises(ValueError, match="версией"):
        runtime.validate_cached_result(result)
