from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aipm3 import latest_runtime as runtime
from aipm3.latest_manual_inputs import (CELEBRITY_VERSION, LEGACY_CELEBRITY_VERSION,
                                       NOTICEABILITY_CELEBRITY_VERSION,
                                       with_celebrity_review, effective_feature_rows,
                                       noticeability_multiplier, recall_multiplier)
from aipm3.latest_interpretation import build_latest_interpretation, public_result
from aipm3.manager_report import report_cards


@pytest.fixture(scope="module")
def rows():
    data = {}
    for task in "nmr":
        history = pd.read_csv(runtime.BUNDLE_DIR / "data" / f"fit_{task}.csv")
        frame = pd.DataFrame([history.iloc[0].to_dict() for _ in range(3)])
        frame["record"] = "manual-flag-test"
        frame["family"] = "unseen-manual-flag-test"
        frame["sha"] = "a" * 64
        frame["brand"] = "Avito"
        frame["vertical"] = "Goods"
        frame["repeat"] = [1, 2, 3]
        frame["phys__duration"] = 30.
        data[task] = frame
    data["n"]["is_celeb"] = 0
    return data


@pytest.fixture(params=["finished", "neuromatics"], scope="module")
def result(rows, request):
    return runtime.score_feature_rows(rows, metadata={"source_sha": "a" * 64},
                                      material_kind=request.param, repeat_count=3)


def test_manual_presence_replaces_detector_and_scales_recall_once(result):
    before = deepcopy(result)
    adjusted = with_celebrity_review(result, True)
    assert result == before
    assert adjusted["feature_rows"] == result["feature_rows"]
    assert adjusted["scores"]["norm_ad_recall"] == pytest.approx(result["scores"]["norm_ad_recall"] * 1.3)
    assert adjusted["scores"]["message_delivery"] == result["scores"]["message_delivery"]
    assert with_celebrity_review(adjusted, True) == adjusted
    restored = with_celebrity_review(adjusted, False)
    assert restored["scores"] == result["scores"]
    effective = effective_feature_rows(adjusted["feature_rows"], adjusted)
    assert all(r["is_celeb"] == 1 for r in effective["n"])
    expected_n = runtime.load_models(result["material_kind"]).heads["n"].predict(pd.DataFrame(effective["n"]))
    np.testing.assert_allclose([p["noticeability"] for p in adjusted["per_repeat"]], expected_n)
    for p in adjusted["per_repeat"]:
        assert p["OPM"] == pytest.approx(p["noticeability"] * p["message_delivery"])
        assert p["Q"] == pytest.approx(p["noticeability"] * p["message_delivery"] * p["norm_ad_recall"])


def legacy_export(current, version):
    """Historical arithmetic, independently of the current multiplier helpers."""
    legacy = deepcopy(current)
    for metadata in (legacy, legacy["metadata"]):
        metadata["celebrity_review"]["version"] = version
    if not legacy["celebrity_review"]["present"]:
        return legacy
    factors = {"norm_ad_recall": 1.2 / 1.3, "Q": 1.2 / 1.3}
    if version == NOTICEABILITY_CELEBRITY_VERSION:
        factors.update(noticeability=1.3, OPM=1.3, Q=1.2)
    for name, factor in factors.items():
        legacy["scores"][name] *= factor
        for row in legacy["per_repeat"]:
            row[name] *= factor
    return legacy


@pytest.mark.parametrize("version", [LEGACY_CELEBRITY_VERSION, NOTICEABILITY_CELEBRITY_VERSION])
@pytest.mark.parametrize("present", [False, True])
def test_legacy_manual_result_is_validated_then_upgraded_without_compounding(result, version, present):
    current = with_celebrity_review(result, present)
    legacy = legacy_export(current, version)
    before = deepcopy(legacy)
    upgraded = runtime.validate_cached_result(legacy)
    assert legacy == before
    assert upgraded["celebrity_review"]["version"] == CELEBRITY_VERSION
    assert upgraded["scores"] == pytest.approx(current["scores"])
    assert runtime.validate_cached_result(upgraded)["scores"] == upgraded["scores"]
    wrong = deepcopy(legacy)
    wrong["scores"]["norm_ad_recall"] *= 1.3
    with pytest.raises(ValueError, match="оценки"):
        runtime.validate_cached_result(wrong)


def test_only_recall_gets_current_bonus_and_old_versions_keep_original_math(result):
    assert noticeability_multiplier(result) == 1.0
    assert noticeability_multiplier(dict(result, celebrity_review=None)) == 1.0
    assert noticeability_multiplier(with_celebrity_review(result, False)) == 1.0
    adjusted = with_celebrity_review(result, True)
    assert noticeability_multiplier(adjusted) == 1.0
    assert recall_multiplier(adjusted) == 1.3
    assert recall_multiplier(with_celebrity_review(result, False)) == 1.0
    first = legacy_export(adjusted, LEGACY_CELEBRITY_VERSION)
    second = legacy_export(adjusted, NOTICEABILITY_CELEBRITY_VERSION)
    assert (recall_multiplier(first), noticeability_multiplier(first)) == (1.2, 1.0)
    assert (recall_multiplier(second), noticeability_multiplier(second)) == (1.2, 1.3)


@pytest.mark.parametrize("repeat_count", [3, 5, 10])
@pytest.mark.parametrize("kind", ["finished", "neuromatics"])
@pytest.mark.parametrize("version", [LEGACY_CELEBRITY_VERSION, NOTICEABILITY_CELEBRITY_VERSION])
def test_recall_replaces_existing_bonus_for_all_run_lengths(rows, repeat_count, kind, version):
    data = {task: pd.DataFrame([frame.iloc[0].to_dict() for _ in range(repeat_count)])
            for task, frame in rows.items()}
    for frame in data.values():
        frame["repeat"] = range(1, repeat_count + 1)
    legacy = runtime.score_feature_rows(data, metadata={"source_sha": "a" * 64,
        "celebrity_review": {"version": version, "present": True,
                             "source_sha": "a" * 64, "origin": "user_declared"}},
        material_kind=kind, repeat_count=repeat_count)
    adjusted = with_celebrity_review(legacy, True)
    old_noticeability_bonus = 1.3 if version == NOTICEABILITY_CELEBRITY_VERSION else 1.0
    ratios = {"noticeability": 1 / old_noticeability_bonus, "OPM": 1 / old_noticeability_bonus,
              "message_delivery": 1., "norm_ad_recall": 1.3 / 1.2,
              "Q": 1.3 / 1.2 / old_noticeability_bonus}
    for name, ratio in ratios.items():
        assert adjusted["scores"][name] == pytest.approx(legacy["scores"][name] * ratio)
        np.testing.assert_allclose([p[name] for p in adjusted["per_repeat"]],
                                   [p[name] * ratio for p in legacy["per_repeat"]])
    assert runtime.validate_cached_result(legacy)["scores"] == adjusted["scores"]
    assert runtime.validate_cached_result(adjusted)["scores"] == adjusted["scores"]


def test_export_reload_keeps_flag_and_rejects_cross_video_or_changed_scores(result):
    adjusted = with_celebrity_review(result, True)
    portable = json.loads(json.dumps(public_result(adjusted, {})))
    assert runtime.validate_cached_result(portable)["scores"] == adjusted["scores"]
    wrong = deepcopy(portable)
    wrong["celebrity_review"]["source_sha"] = "b" * 64
    with pytest.raises(ValueError, match="медийной персоне"):
        runtime.validate_cached_result(wrong)
    wrong = deepcopy(portable)
    wrong["scores"]["norm_ad_recall"] *= 1.2
    with pytest.raises(ValueError, match="оценки"):
        runtime.validate_cached_result(wrong)
    wrong = deepcopy(portable)
    wrong["celebrity_review"]["version"] = NOTICEABILITY_CELEBRITY_VERSION
    with pytest.raises(ValueError, match="оценки"):
        runtime.validate_cached_result(wrong)
    wrong = deepcopy(portable)
    wrong["celebrity_review"]["version"] = "unsupported-policy"
    with pytest.raises(ValueError, match="медийной персоне"):
        runtime.validate_cached_result(wrong)


def test_interpretation_matches_adjusted_scores_without_media_person_output(result):
    adjusted = with_celebrity_review(result, True)
    explanation = build_latest_interpretation(adjusted)
    cards = {c["task"]: c for c in explanation["cards"]}
    for task, name in runtime.SCORE_NAMES.items():
        assert cards[task]["score"] == pytest.approx(adjusted["scores"][name])
        assert explanation["details"][task]["additivity_error"] < 1e-9
        predictions = [row[name] for row in adjusted["per_repeat"]]
        np.testing.assert_allclose(explanation["details"][task]["per_repeat_scores"], predictions)
        np.testing.assert_allclose(cards[task]["repeat_index_range"],
                                   np.quantile(predictions, [.1, .9])
                                   * cards[task]["index"] / cards[task]["score"])
    assert explanation["overall"]["score"] == adjusted["scores"]["Q"]
    recall = next(c for c in report_cards(explanation) if c["task"] == "r")
    assert not any(d["feature"] == "manual_celebrity" for d in recall["strengths"])
    assert "manual_celebrity" not in json.dumps(recall, ensure_ascii=False)


def test_checkbox_updates_cached_result_without_calls_and_survives_reopen(result, tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    from aipm3 import latest_pipeline
    import sys
    sys.modules.pop("app_pages.latest_pretest", None)
    kind = result["material_kind"]
    env = "AIPM_LATEST_RESULT_JSON" if kind == "finished" else "AIPM_NEUROMATICS_RESULT_JSON"
    preset = tmp_path / "result.json"
    preset.write_text(json.dumps(result))
    monkeypatch.setenv(env, str(preset))
    monkeypatch.setattr(latest_pipeline, "run_latest_analysis", lambda **kw: pytest.fail("Unexpected analysis"))
    page = "latest_pretest.py" if kind == "finished" else "neuromatics_pretest.py"
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app_pages" / page)).run(timeout=45)
    assert not app.exception
    key = f"latest_{kind}_celebrity_" + result["source_sha"]
    assert app.checkbox(key=key).value is False
    app.checkbox(key=key).check().run(timeout=45)
    assert not app.exception
    adjusted = app.session_state[f"latest_{kind}_result"]
    assert adjusted["scores"]["norm_ad_recall"] == pytest.approx(result["scores"]["norm_ad_recall"] * 1.3)
    assert not any("медийн" in x.value for x in [*app.success, *app.markdown])
    app.run(timeout=45)
    assert app.session_state[f"latest_{kind}_result"]["scores"] == adjusted["scores"]
    app.checkbox(key=key).uncheck().run(timeout=45)
    assert not app.exception
    assert app.session_state[f"latest_{kind}_result"]["scores"] == result["scores"]


@pytest.mark.parametrize("version", [LEGACY_CELEBRITY_VERSION, NOTICEABILITY_CELEBRITY_VERSION])
def test_existing_session_migrates_both_old_policies_without_analysis(result, version, monkeypatch):
    from streamlit.testing.v1 import AppTest
    from aipm3 import latest_pipeline
    import sys
    sys.modules.pop("app_pages.latest_pretest", None)
    monkeypatch.delenv("AIPM_LATEST_RESULT_JSON", raising=False)
    monkeypatch.delenv("AIPM_NEUROMATICS_RESULT_JSON", raising=False)
    monkeypatch.setattr(latest_pipeline, "run_latest_analysis", lambda **kw: pytest.fail("Unexpected analysis"))
    kind = result["material_kind"]
    current = with_celebrity_review(result, True)
    page = "latest_pretest.py" if kind == "finished" else "neuromatics_pretest.py"
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app_pages" / page))
    app.session_state[f"latest_{kind}_result"] = legacy_export(current, version)
    app.run(timeout=45)
    assert not app.exception
    migrated = app.session_state[f"latest_{kind}_result"]
    assert migrated["celebrity_review"]["version"] == CELEBRITY_VERSION
    assert migrated["scores"] == pytest.approx(current["scores"])
    app.run(timeout=45)
    assert not app.exception
    assert app.session_state[f"latest_{kind}_result"]["scores"] == migrated["scores"]
