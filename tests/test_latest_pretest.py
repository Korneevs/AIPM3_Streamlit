"""Offline UI verifies explicit live action and model-association copy."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest
from streamlit.testing.v1 import AppTest

from aipm3 import latest_pipeline, latest_interpretation, latest_runtime


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def isolate_page_import(monkeypatch):
    # The wrapper imports this module; refresh it so each AppTest receives its own mocked dependencies.
    sys.modules.pop("app_pages.latest_pretest", None)
    monkeypatch.delenv("AIPM_NEUROMATICS_RESULT_JSON", raising=False)


def interpretation():
    driver = dict(feature="fresh__audiovisual_claim_alignment", value=3, label="Связь изображения и озвучки", evidence={"verified": True,
        "observation": "Изображение показывает заявленное действие.",
        "episodes": [{"start": 3., "end": 5.}]},
        index_points=5, usable=True, direction="limits", interpretation_kind="association_only", why="Это связь в обученной модели.", check=None)
    return dict(cards=[dict(title=title, index=index, level="Типичный уровень",
                           repeat_index_range=[95., 105.], strengths=[], limitations=[driver], unresolved=[])
                      for title, index in [("Заметность", 101), ("Считываемость", 92), ("Запоминаемость", 98)]],
                scale_note="100 — средняя оценка исторических роликов; не процент зрителей.",
                repeat_note="Разброс повторов не является доверительным интервалом.",
                interpretation_note="Отдельную правку нужно проверить на новой версии.", evidence_runs=3)


@pytest.mark.parametrize("repeat_count", [3, 10])
def test_result_page_shows_three_indices_and_preserves_associations(repeat_count):
    source = (ROOT / "app_pages/latest_pretest.py").read_text().split('\nif __name__ in {')[0]
    app = AppTest.from_string(source + '\nshow_latest_result(st.session_state["result"], st.session_state["interpretation"])\n')
    result = {"scores": {"Q": .04, "OPM": .2}, "scoring_version": latest_runtime.SCORING_VERSION}
    result["repeat_count"] = repeat_count
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
    assert "Что нельзя уверенно объяснить" in text
    assert "Изображение показывает заявленное действие." not in text
    assert "На что обратить внимание" not in text
    assert "100 - средняя оценка" in captions
    for forbidden in ["Сравнение и разброс", "Версия расчёта", "пункта индекса", "SHAP", "исходные признаки", "Q", "OPM", "Особенность модели", "замороженная"]:
        assert forbidden not in text + captions
    assert not app.expander
    assert len(app.get("plotly_chart")) == 0
    assert "#FFF6D6" in text and "В норме" in text
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
    result = {"scores": {"Q": .04, "OPM": .2}, "source_name": "cached.mp4",
              "scoring_version": latest_runtime.SCORING_VERSION}
    saved = tmp_path / "result.json"
    saved.write_text(json.dumps(result))
    monkeypatch.setenv("AIPM_LATEST_RESULT_JSON", str(saved))
    monkeypatch.delenv("VSELLM_API_KEY", raising=False)
    monkeypatch.setattr(latest_pipeline, "run_latest_analysis", lambda **_: pytest.fail("Unexpected analysis"))
    monkeypatch.setattr(latest_runtime, "validate_cached_result", lambda result, **_: result)
    monkeypatch.setattr(latest_interpretation, "build_latest_interpretation", lambda *_, **__: interpretation())
    app = AppTest.from_file(str(ROOT / "app_pages/latest_pretest.py")).run(timeout=30)
    assert not app.exception
    assert [item.value for item in app.metric] == ["101", "92", "98"]
    assert any("cached.mp4" in item.value for item in app.caption)


@pytest.mark.parametrize("kind,page", [
    ("finished", "latest_pretest.py"),
    ("neuromatics", "neuromatics_pretest.py"),
])
def test_other_material_result_is_not_displayed(kind, page, monkeypatch):
    monkeypatch.delenv("AIPM_LATEST_RESULT_JSON", raising=False)
    monkeypatch.delenv("AIPM_NEUROMATICS_RESULT_JSON", raising=False)
    monkeypatch.setattr(latest_pipeline, "run_latest_analysis", lambda **_: pytest.fail("Unexpected analysis"))
    other = "neuromatics" if kind == "finished" else "finished"
    app = AppTest.from_file(str(ROOT / "app_pages" / page))
    app.session_state[f"latest_{other}_result"] = {
        "material_kind": other, "scoring_version": latest_runtime.scoring_version_for(other),
        "scores": {"Q": .04, "OPM": .2}, "source_name": "other-material.mp4",
    }
    app.run(timeout=30)
    assert not app.exception
    assert app.title[0].value == latest_runtime.MATERIAL_LABELS[kind]
    assert not app.metric
    assert not any("other-material.mp4" in caption.value for caption in app.caption)
    assert app.radio[0].key == f"latest_{kind}_source"
    assert len(app.button) == 2


def test_neuromatics_does_not_read_finished_preset(tmp_path, monkeypatch):
    saved = tmp_path / "finished.json"
    saved.write_text(json.dumps({"scoring_version": latest_runtime.SCORING_VERSION}))
    monkeypatch.setenv("AIPM_LATEST_RESULT_JSON", str(saved))
    monkeypatch.delenv("AIPM_NEUROMATICS_RESULT_JSON", raising=False)
    monkeypatch.setattr(latest_pipeline, "run_latest_analysis", lambda **_: pytest.fail("Unexpected analysis"))
    monkeypatch.setattr(latest_runtime, "validate_cached_result", lambda **_: pytest.fail("Read wrong preset"))
    app = AppTest.from_file(str(ROOT / "app_pages/neuromatics_pretest.py")).run()
    assert not app.exception
    assert not app.metric
    assert not app.error


@pytest.mark.parametrize("kind,page", [
    ("finished", "latest_pretest.py"),
    ("neuromatics", "neuromatics_pretest.py"),
])
def test_saved_result_of_wrong_material_is_rejected(kind, page, monkeypatch):
    from io import BytesIO
    import streamlit as st

    monkeypatch.delenv("AIPM_LATEST_RESULT_JSON", raising=False)
    monkeypatch.delenv("AIPM_NEUROMATICS_RESULT_JSON", raising=False)
    other = "neuromatics" if kind == "finished" else "finished"
    saved = BytesIO(json.dumps({
        "material_kind": other, "scoring_version": latest_runtime.scoring_version_for(other),
        "scores": {"Q": .04, "OPM": .2},
    }).encode())
    monkeypatch.setattr(st, "file_uploader", lambda *_, **__: saved)
    monkeypatch.setattr(latest_pipeline, "run_latest_analysis", lambda **_: pytest.fail("Unexpected analysis"))
    app = AppTest.from_file(str(ROOT / "app_pages" / page))
    app.session_state[f"latest_{kind}_source"] = "Открыть сохранённый результат"
    app.run(timeout=30)
    assert not app.exception
    assert len(app.error) == 1
    assert not app.metric
    assert f"latest_{kind}_result" not in app.session_state


@pytest.mark.parametrize("kind,page", [
    ("finished", "latest_pretest.py"),
    ("neuromatics", "neuromatics_pretest.py"),
])
def test_live_action_passes_material_kind_and_uses_three_runs(kind, page, monkeypatch):
    from io import BytesIO
    import streamlit as st

    monkeypatch.delenv("AIPM_LATEST_RESULT_JSON", raising=False)
    monkeypatch.delenv("AIPM_NEUROMATICS_RESULT_JSON", raising=False)
    uploaded = BytesIO(b"placeholder-video")
    uploaded.name = "clip.mp4"
    monkeypatch.setattr(st, "file_uploader", lambda *_, **__: uploaded)
    calls = []

    def analyze(**kwargs):
        calls.append(kwargs)
        return {"material_kind": kind, "scoring_version": latest_runtime.scoring_version_for(kind),
                "scores": {"Q": .04, "OPM": .2}}

    monkeypatch.setattr(latest_pipeline, "run_latest_analysis", analyze)
    monkeypatch.setattr(latest_interpretation, "build_latest_interpretation", lambda *_, **__: interpretation())
    app = AppTest.from_file(str(ROOT / "app_pages" / page))
    app.secrets["VSELLM_API_KEY"] = "test-key"
    app.run(timeout=30)
    assert not app.exception
    assert not calls
    assert any("Анализ и проверка наблюдений" in caption.value for caption in app.caption)
    app.button(key=f"latest_{kind}_live").click().run(timeout=30)
    assert not app.exception
    assert len(calls) == 1
    assert calls[0]["material_kind"] == kind
    assert calls[0]["repeat_count"] == 3
    assert calls[0]["allow_live"] is True
    assert app.session_state[f"latest_{kind}_result"]["source_name"] == "clip.mp4"
    assert len(app.metric) == 3
