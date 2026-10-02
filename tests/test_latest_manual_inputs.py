from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aipm3 import latest_runtime as runtime
from aipm3.latest_manual_inputs import with_celebrity_review, effective_feature_rows
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
    assert adjusted["scores"]["norm_ad_recall"] == pytest.approx(result["scores"]["norm_ad_recall"] * 1.2)
    assert adjusted["scores"]["message_delivery"] == result["scores"]["message_delivery"]
    assert with_celebrity_review(adjusted, True) == adjusted
    restored = with_celebrity_review(adjusted, False)
    assert restored["scores"] == result["scores"]
    effective = effective_feature_rows(adjusted["feature_rows"], adjusted)
    assert all(r["is_celeb"] == 1 for r in effective["n"])
    expected_n = runtime.load_models(result["material_kind"]).heads["n"].predict(pd.DataFrame(effective["n"]))
    np.testing.assert_allclose([p["noticeability"] for p in adjusted["per_repeat"]], expected_n)
    for p in adjusted["per_repeat"]:
        assert p["Q"] == pytest.approx(p["noticeability"] * p["message_delivery"] * p["norm_ad_recall"])


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


def test_interpretation_matches_adjusted_scores_and_shows_positive_reason(result):
    adjusted = with_celebrity_review(result, True)
    explanation = build_latest_interpretation(adjusted)
    cards = {c["task"]: c for c in explanation["cards"]}
    for task, name in runtime.SCORE_NAMES.items():
        assert cards[task]["score"] == pytest.approx(adjusted["scores"][name])
        assert explanation["details"][task]["additivity_error"] < 1e-9
    assert explanation["overall"]["score"] == adjusted["scores"]["Q"]
    recall = next(c for c in report_cards(explanation) if c["task"] == "r")
    manual = next(d for d in recall["strengths"] if d["feature"] == "manual_celebrity")
    assert "положительно влияет" in manual["takeaway"]
    assert "1.2" not in json.dumps(recall, ensure_ascii=False)


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
    assert adjusted["scores"]["norm_ad_recall"] == pytest.approx(result["scores"]["norm_ad_recall"] * 1.2)
    assert any("положительно влияет" in x.value for x in app.markdown)
    app.run(timeout=45)
    assert app.session_state[f"latest_{kind}_result"]["scores"] == adjusted["scores"]
    app.checkbox(key=key).uncheck().run(timeout=45)
    assert not app.exception
    assert app.session_state[f"latest_{kind}_result"]["scores"] == result["scores"]
