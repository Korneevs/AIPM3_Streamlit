"""Offline UI verifies explicit live action and model-association copy."""
from copy import deepcopy
import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from aipm3 import latest_pipeline, latest_interpretation, latest_runtime


ROOT = Path(__file__).resolve().parents[1]


def interpretation():
    driver = dict(label="Связь изображения и озвучки", evidence={
        "observation": "Изображение показывает заявленное действие.",
        "episodes": [{"start": 3., "end": 5.}]},
        interpretation_kind="association_only", why="Это связь в обученной модели.", check=None)
    return dict(cards=[dict(title=title, index=index, level="Типичный уровень",
                           repeat_index_range=[95., 105.], strengths=[], limitations=[driver], unresolved=[])
                      for title, index in [("Заметность", 101), ("Считываемость", 92), ("Запоминаемость", 98)]],
                scale_note="100 — средняя оценка исторических роликов; не процент зрителей.",
                repeat_note="Разброс повторов не является доверительным интервалом.",
                interpretation_note="Отдельную правку нужно проверить на новой версии.", evidence_runs=3)


def test_result_page_shows_three_indices_and_preserves_associations():
    source = (ROOT / "app_pages/latest_pretest.py").read_text().split('\nif __name__ == "__main__":')[0]
    app = AppTest.from_string(source + '\nshow_latest_result(st.session_state["result"], st.session_state["interpretation"])\n')
    result = {"scores": {"Q": .04, "OPM": .2}}
    content = interpretation()
    before = deepcopy(content)
    app.session_state["result"] = result
    app.session_state["interpretation"] = content
    app.run(timeout=30)
    assert not app.exception
    assert [item.label for item in app.metric] == ["Заметность", "Считываемость", "Запоминаемость"]
    assert [item.value for item in app.metric] == ["101", "92", "98"]
    text = " ".join(item.value for item in app.markdown)
    captions = " ".join(item.value for item in app.caption)
    assert "Что модель учла в минус" in text
    assert "Особенность модели, не рекомендация" in captions
    assert "Что проверить в следующей версии" not in text
    assert "не процент зрителей" in captions
    assert len(app.get("plotly_chart")) == 1
    assert len(app.get("download_button")) == 1
    assert app.session_state["interpretation"] == before


def test_loading_local_page_and_missing_upload_never_dispatch_live_calls(monkeypatch):
    monkeypatch.delenv("AIPM_LATEST_RESULT_JSON", raising=False)
    monkeypatch.setattr(latest_pipeline, "run_latest_analysis", lambda **_: pytest.fail("Unexpected analysis"))
    app = AppTest.from_file(str(ROOT / "app_pages/latest_pretest.py")).run()
    assert not app.exception
    assert len(app.button) == 2
    app.button[1].click().run()
    assert not app.exception
    assert app.warning[0].value == "Сначала загрузите ролик."


def test_offline_saved_result_does_not_require_api_key(tmp_path, monkeypatch):
    result = {"scores": {"Q": .04, "OPM": .2}, "source_name": "cached.mp4"}
    saved = tmp_path / "result.json"
    saved.write_text(json.dumps(result))
    monkeypatch.setenv("AIPM_LATEST_RESULT_JSON", str(saved))
    monkeypatch.delenv("VSELLM_API_KEY", raising=False)
    monkeypatch.setattr(latest_pipeline, "run_latest_analysis", lambda **_: pytest.fail("Unexpected analysis"))
    monkeypatch.setattr(latest_runtime, "validate_cached_result", lambda result: result)
    monkeypatch.setattr(latest_interpretation, "build_latest_interpretation", lambda *_, **__: interpretation())
    app = AppTest.from_file(str(ROOT / "app_pages/latest_pretest.py")).run(timeout=30)
    assert not app.exception
    assert [item.value for item in app.metric] == ["101", "92", "98"]
    assert any("cached.mp4" in item.value for item in app.caption)
