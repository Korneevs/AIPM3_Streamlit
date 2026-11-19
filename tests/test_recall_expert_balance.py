"""Offline checks of the explicit recall policy and additive UI explanation."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from aipm3 import latest_runtime as runtime
from aipm3.latest_manual_inputs import with_recall_celebrity_review, with_celebrity_review
from aipm3.latest_interpretation import build_latest_interpretation
from aipm3.latest_balance_ui import balance_items, balance_figure


@pytest.fixture(params=["finished", "neuromatics"], scope="module")
def result(request):
    fixtures = json.loads((Path(__file__).parent / "fixtures/aipm0719.json").read_text())
    fixture = next(f for f in fixtures if f["material_kind"] == request.param and f["repeat_count"] == 5)
    return runtime.score_feature_rows(fixture["frames"],
        metadata={"source_sha": fixture["frames"]["n"][0]["sha"]},
        material_kind=request.param, repeat_count=5)


def test_recall_policy_is_explicit_idempotent_and_source_bound(result):
    before = deepcopy(result)
    adjusted = with_recall_celebrity_review(result, True)
    assert result == before
    assert adjusted["feature_rows"] == result["feature_rows"]
    for name in ("noticeability", "message_delivery", "OPM"):
        assert adjusted["scores"][name] == result["scores"][name]
    assert adjusted["scores"]["norm_ad_recall"] == pytest.approx(result["scores"]["norm_ad_recall"] * 1.2)
    assert adjusted["scores"]["Q"] == pytest.approx(result["scores"]["Q"] * 1.2 ** result["q_exponents"][2])
    assert with_recall_celebrity_review(adjusted, True) == adjusted
    assert with_recall_celebrity_review(adjusted, False)["scores"] == result["scores"]
    assert runtime.validate_cached_result(json.loads(json.dumps(adjusted)))["scores"] == adjusted["scores"]
    wrong = deepcopy(adjusted)
    wrong["recall_celebrity_review"]["source_sha"] = "invalid-video"
    with pytest.raises(ValueError, match="селебрити"):
        runtime.validate_cached_result(wrong)
    # An old noticeability flag must not silently enable the new recall policy.
    assert with_celebrity_review(result, True)["scores"]["norm_ad_recall"] == result["scores"]["norm_ad_recall"]


def test_expert_bar_preserves_learned_contributions_and_classifies_adjusted_score(result):
    before = build_latest_interpretation(result)
    adjusted = with_recall_celebrity_review(result, True)
    after = build_latest_interpretation(adjusted)
    for task in "nmr":
        np.testing.assert_allclose(after["details"][task]["per_repeat_shap"],
                                   before["details"][task]["per_repeat_shap"], rtol=0, atol=0)
        detail = after["details"][task]
        assert detail["additivity_error"] < 1e-10
        items = balance_items(after, task)
        actual = sum(i["value"] for i in items if i["measure"] != "total")
        assert actual == pytest.approx(detail["actual"])
        card = next(c for c in after["cards"] if c["task"] == task)
        assert card["class"] == adjusted["classes"][task]
        assert card["score"] == pytest.approx(adjusted["scores"][runtime.SCORE_NAMES[task]])
        assert len([i for i in items if i.get("expert")]) == (1 if task == "r" else 0)
        figure = balance_figure(after, task)
        assert figure.data[0].type == "waterfall"
    assert after["overall"]["class"] == adjusted["classes"]["q"]


def test_recall_checkbox_updates_and_restores_cached_result_offline(result, tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    from aipm3 import latest_pipeline
    import sys
    sys.modules.pop("app_pages.latest_pretest", None)
    kind = result["material_kind"]
    env = "AIPM_LATEST_RESULT_JSON" if kind == "finished" else "AIPM_NEUROMATICS_RESULT_JSON"
    preset = tmp_path / "result.json"
    preset.write_text(json.dumps(result))
    monkeypatch.setenv(env, str(preset))
    monkeypatch.setattr(latest_pipeline, "run_latest_analysis", lambda **kw: pytest.fail("Unexpected API call"))
    page = "latest_pretest.py" if kind == "finished" else "neuromatics_pretest.py"
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app_pages" / page))
    app.run(timeout=45)
    assert not app.exception
    key = f"latest_{kind}_recall_celebrity_" + result["source_sha"]
    assert app.checkbox(key=key).value is False
    app.checkbox(key=key).check().run(timeout=45)
    assert not app.exception
    adjusted = app.session_state[f"latest_{kind}_result"]
    assert adjusted["scores"] == with_recall_celebrity_review(result, True)["scores"]
    app.run(timeout=45)
    assert app.session_state[f"latest_{kind}_result"]["scores"] == adjusted["scores"]
    app.checkbox(key=key).uncheck().run(timeout=45)
    assert not app.exception
    assert app.session_state[f"latest_{kind}_result"]["scores"] == result["scores"]
    # Import another saved state for the identical SHA: old widget state must
    # not silently override its manual policy in either direction.
    yes_export = tmp_path / "yes.json"
    yes_export.write_text(json.dumps(with_recall_celebrity_review(result, True)))
    monkeypatch.setenv(env, str(yes_export))
    app.run(timeout=45)
    assert not app.exception
    assert app.checkbox(key=key).value is True
    monkeypatch.setenv(env, str(preset))
    app.run(timeout=45)
    assert not app.exception
    assert app.checkbox(key=key).value is False
    assert app.session_state[f"latest_{kind}_result"]["scores"] == result["scores"]


def test_unavailable_observation_fingerprint_matches_exported_null():
    from aipm3.neuromatics_manager import draft_fingerprint
    assert draft_fingerprint({"observation": float("nan")}) == draft_fingerprint({"observation": None})
