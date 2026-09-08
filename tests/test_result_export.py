"""Retired campaign-brief diagnostics must not escape in exported results."""
import copy
import json
from pathlib import Path

from aipm3.result_export import export_result


def test_export_removes_retired_fields_without_mutating_source():
    result = {
        "aipm1": {"raw_score": 1}, "aipm2": {"raw_score": 2},
        "message_delivery": {"raw_score": .3}, "aipm3": {"index_100": 90},
        "main_idea": "Выбор автомобилей", "objective_features": {"feature": 1},
        "interpretation": {"strengths": ["old"]}, "manager_readout": {"pillars": "old"},
        "objective_runs": ["private"], "transcripts": ["private"], "prepared_video": "private",
        "brief": {"uvp": "old"}, "uvp": "old", "rtb": "old",
        "diagnostics": {
            "current_brief": {"uvp": "old", "rtb": "old"},
            "brief_alignment": {"secret_brief": "old"}, "message_alignment": {"uvp": "old"},
            "brief_details": {"rtb": "old"}, "brief_details_error": "old",
            "scene_evidence": {"scenes": [{"quote": "Выбор автомобилей"}]},
        },
    }
    before = copy.deepcopy(result)
    exported = export_result(result)
    assert result == before
    for key in ["aipm1", "aipm2", "message_delivery", "aipm3", "main_idea", "objective_features"]:
        assert exported[key] == result[key]
    assert exported["diagnostics"] == {"scene_evidence": result["diagnostics"]["scene_evidence"]}
    serialized = json.dumps(exported)
    assert "uvp" not in serialized and "rtb" not in serialized and "brief" not in serialized
    assert "private" not in serialized
    assert 'interpretation' not in exported and 'manager_readout' not in exported
    exported["aipm1"]["raw_score"] = 999
    exported["diagnostics"]["scene_evidence"]["scenes"].clear()
    assert result == before


def test_empty_diagnostics_and_clean_result_export():
    assert export_result({"diagnostics": {"current_brief": {"uvp": "x"}}}) == {}
    assert export_result({"aipm3": {"index_100": 100}}) == {"aipm3": {"index_100": 100}}


def test_app_has_no_reachable_brief_ui():
    root = Path(__file__).resolve().parents[1]
    page = (root / "app_pages/video_pretest.py").read_text()
    ui = (root / "aipm3/review_ui.py").read_text()
    for source in [page, ui]:
        assert "show_brief_review" not in source
        assert "message_review_ui" not in source
    assert "export_result(result)" in page
