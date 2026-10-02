"""Offline production-mode regressions using only the bundled historical schemas."""
from copy import deepcopy
import importlib.util
import json

import numpy as np
import pandas as pd
import pytest

from aipm3 import latest_runtime as runtime
from aipm3 import neuromatics_models as neuro
from aipm3.latest_interpretation import public_result


@pytest.fixture(scope="module")
def rows():
    frames = {}
    for task in "nmr":
        history = pd.read_csv(runtime.BUNDLE_DIR / "data" / f"fit_{task}.csv",
                              float_precision="round_trip")
        frame = pd.DataFrame([history.iloc[0].to_dict() for _ in range(10)])
        frame["record"] = "runtime-neuromatics"
        frame["family"] = "unseen-runtime-family"
        frame["brand"] = "Avito"
        frame["vertical"] = "Goods"
        frame["sha"] = "b" * 64
        frame["repeat"] = np.arange(1, 11)
        frame["phys__duration"] = 30.
        frame["total_video_duration_sec"] = 30.
        frames[task] = frame
    frames["n"]["phys__audio_dynamic_range_db"] = 50.
    frames["n"]["phys__motion_mean"] = 0.
    frames["n"]["state_transformation_present"] = 1.
    frames["n"]["state_transformation"] = 0.
    frames["r"]["jingle_present"] = [1.] * 6 + [0.] * 4
    return frames


def test_modes_use_selected_heads_without_mutating_inputs(rows):
    before = {task: frame.copy(deep=True) for task, frame in rows.items()}
    result = runtime.score_feature_rows(rows, material_kind="neuromatics")
    repeated = pd.DataFrame(result["per_repeat"])
    # Compare the public runtime to the independently imported portable scorer.
    spec = importlib.util.spec_from_file_location("_portable_recall_test", neuro.BUNDLE_DIR / "predict.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    portable = module.NeuroRecall(neuro.BUNDLE_DIR / "parameters.json")
    np.testing.assert_array_equal(repeated.norm_ad_recall, portable.predict(rows["r"]))
    expected_n = rows["n"].copy()
    expected_n["phys__audio_dynamic_range_db"] = 19.09761743545532
    expected_n["phys__motion_mean"] = .0160984480753541
    expected_n["state_transformation_present"] = 0.
    frozen = runtime.load_latest_models()
    np.testing.assert_array_equal(repeated.noticeability, frozen.heads["n"].predict(expected_n))
    np.testing.assert_array_equal(repeated.message_delivery, frozen.heads["m"].predict(rows["m"]))
    np.testing.assert_array_equal(repeated.Q, repeated.noticeability * repeated.message_delivery * repeated.norm_ad_recall)
    assert result["scores"]["Q"] == pytest.approx(repeated.Q.mean(), abs=1e-14)
    assert [len(result["model_inputs"][task]) for task in "nmr"] == [9, 9, 7]
    finished = runtime.score_feature_rows(rows)
    expected = frozen.score(*(rows[task] for task in "nmr")).iloc[0]
    for name, score in finished["scores"].items():
        assert score == expected[name]
    for task in "nmr":
        pd.testing.assert_frame_equal(rows[task], before[task])
    assert all(row["state_transformation_present"] == 1. for row in result["feature_rows"]["n"])
    assert all(row["state_transformation"] == 0. for row in result["feature_rows"]["n"])


@pytest.mark.parametrize("mode", ["finished", "neuromatics"])
def test_three_runs_use_unchanged_heads_and_require_explicit_count(rows, mode):
    short = {task: frame.iloc[:3].copy() for task, frame in rows.items()}
    short["n"]["phys__motion_mean"] = [.01, .04, .10]
    short["r"]["brand_first_mention_seconds"] = [1., 5., 15.]
    model = runtime.load_models(mode)
    expected = {task: model.heads[task].predict(short[task]) for task in "nmr"}
    result = runtime.score_feature_rows(short, material_kind=mode, repeat_count=3)
    for task, name in runtime.SCORE_NAMES.items():
        np.testing.assert_array_equal(pd.DataFrame(result["per_repeat"])[name], expected[task])
    assert result["scores"]["Q"] == pytest.approx(np.mean(expected["n"] * expected["m"] * expected["r"]), abs=1e-14)
    assert result["model_sha256"] == runtime.artifact_hashes_for(mode)
    assert result["repeat_count"] == 3
    assert runtime.validate_cached_result(result)["scores"] == result["scores"]
    from aipm3.latest_interpretation import build_latest_interpretation
    explanation = build_latest_interpretation(result)
    for task, details in explanation["details"].items():
        assert len(details["per_repeat_scores"]) == 3
        assert details["actual"] == pytest.approx(result["scores"][runtime.SCORE_NAMES[task]], abs=1e-12)
        assert details["additivity_error"] < 1e-9
    with pytest.raises(ValueError):
        runtime.score_feature_rows(short, material_kind=mode)
    for count in (2, 10):
        changed = deepcopy(result)
        changed["repeat_count"] = count
        with pytest.raises(ValueError):
            runtime.validate_cached_result(changed)
    legacy = runtime.score_feature_rows(rows, material_kind=mode)
    legacy.pop("repeat_count")
    assert runtime.validate_cached_result(legacy)["scores"] == legacy["scores"]


@pytest.mark.parametrize("bad", ["missing", "duplicate", "misaligned", "legacy_state"])
def test_neuromatics_requires_ten_aligned_complete_readings(rows, bad):
    changed = {task: frame.copy(deep=True) for task, frame in rows.items()}
    if bad == "missing":
        changed["n"] = changed["n"].iloc[:-1]
    elif bad == "duplicate":
        changed["r"].loc[9, "repeat"] = 9
    elif bad == "misaligned":
        changed["m"].loc[9, "repeat"] = 11
    else:
        changed["n"] = changed["n"].drop(columns="state_transformation")
    with pytest.raises(ValueError):
        runtime.score_feature_rows(changed, material_kind="neuromatics")


@pytest.mark.parametrize("mode", ["finished", "neuromatics"])
def test_saved_result_keeps_mode_identity_and_rejects_incompatible_results(rows, mode):
    result = runtime.score_feature_rows(rows, material_kind=mode)
    before = deepcopy(result)
    exported = public_result(result, {"material_kind": mode})
    assert result == before
    for field in ("material_kind", "scoring_version", "model_sha256", "feature_rows"):
        assert exported[field] == result[field]
    restored = runtime.validate_cached_result(json.loads(json.dumps(exported, allow_nan=False)),
                                             material_kind=mode)
    assert restored["scores"] == result["scores"]
    other = "finished" if mode == "neuromatics" else "neuromatics"
    with pytest.raises(ValueError, match="другому типу"):
        runtime.validate_cached_result(result, material_kind=other)
    for field, value in (("material_kind", other), ("protocol_version", "foreign-protocol"),
                         ("scoring_version", "retired-model")):
        changed = deepcopy(result)
        changed[field] = value
        with pytest.raises(ValueError):
            runtime.validate_cached_result(changed)


def test_live_measurement_conversion_retains_both_state_readings(rows):
    # Reconstruct complete measurement envelopes from shipped feature schemas.
    # The two objective protocols intentionally disagree about before/after.
    measurements = []
    model = runtime.load_latest_models()
    for index in range(10):
        a1 = {"main_character": rows["m"].iloc[index].main_character,
              "promo": rows["m"].iloc[index].promo, "state_transformation": index % 2}
        a2 = {"total_video_duration_sec": 30., "state_transformation_present": 1 - index % 2}
        panel, fresh = {}, {}
        for task in "nmr":
            for feature in model.heads[task].state["columns"]:
                if feature in ("brand_history", "state_transformation_present", "main_character", "promo") or feature.startswith("phys__"):
                    continue
                value = rows[task].iloc[index][feature]
                if feature.startswith("panel__"):
                    panel[feature.removeprefix("panel__")] = value
                elif feature.startswith("fresh__"):
                    fresh[feature.removeprefix("fresh__")] = value
                else:
                    a2[feature] = value
        measurements.append(dict(repeat=index + 1, source_sha="b" * 64,
            objective_features={"aipm1": a1, "aipm2": a2}, fresh=fresh,
            diagnostic_panel=[dict(panel, respondent_id=str(person)) for person in range(30)]))
    before = deepcopy(measurements)
    converted = runtime.rows_from_measurements(measurements, source_sha="b" * 64,
        physical={"phys__audio_dynamic_range_db": 50., "phys__motion_mean": 0.}, duration=30.,
        material_kind="neuromatics")
    assert measurements == before
    assert [row["state_transformation"] for row in converted["n"]] == [i % 2 for i in range(10)]
    assert [row["state_transformation_present"] for row in converted["n"]] == [1 - i % 2 for i in range(10)]
    assert len(runtime.score_feature_rows(converted, material_kind="neuromatics")["per_repeat"]) == 10
