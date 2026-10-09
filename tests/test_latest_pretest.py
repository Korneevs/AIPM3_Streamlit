"""Offline UI verifies explicit live action and model-association copy."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest
from streamlit.testing.v1 import AppTest

from aipm3 import latest_pipeline, latest_interpretation, latest_runtime, latest_manual_inputs


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def isolate_page_import(monkeypatch):
    # The wrapper imports this module; refresh it so each AppTest receives its own mocked dependencies.
    sys.modules.pop("app_pages.latest_pretest", None)
    monkeypatch.delenv("AIPM_NEUROMATICS_RESULT_JSON", raising=False)
    # These page-routing fixtures intentionally contain no feature rows.
    # Real scoring/export and checkbox transitions are covered separately.
    monkeypatch.setattr(latest_manual_inputs, "with_celebrity_review", lambda result, present: result)


def interpretation():
    driver = dict(feature="fresh__audiovisual_claim_alignment", value=3, label="Связь изображения и озвучки", evidence={"verified": True,
        "observation": "Изображение показывает заявленное действие.",
        "episodes": [{"start": 3., "end": 5.}]},
        index_points=-5, contribution=-.05, usable=True, direction="limits", interpretation_kind="association_only", why="Это связь в обученной модели.", check=None)
    return dict(cards=[dict(task=task, title=title, index=index, level="Типичный уровень",
                           repeat_index_range=[95., 105.], strengths=[], limitations=[driver], unresolved=[])
                      for task, title, index in [("n", "Заметность", 101), ("m", "Считываемость", 92), ("r", "Запоминаемость", 98)]],
                details={task: {"drivers": [deepcopy(driver)]} for task in 'nmr'},
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
    assert [tab.label for tab in app.tabs] == ["Заметность", "Запоминаемость", "Считываемость основной идеи"]
    text = " ".join(item.value for item in app.markdown)
    captions = " ".join(item.value for item in app.caption)
    assert "не даёт понятного основания" in text
    assert all(value in text for value in [">+1%<", ">−8%<", ">−2%<"])
    assert "Различаем отсутствие связи" in text
    assert "На что обратить внимание" not in text
    assert "0% - средняя оценка" in captions
    for forbidden in ["Сравнение и разброс", "Версия расчёта", "пункта индекса", "SHAP", "исходные признаки", "Q", "OPM", "Особенность модели", "замороженная"]:
        assert forbidden not in text + captions
    assert not app.expander
    assert not app.get("plotly_chart")
    assert "Вес среди" not in text
    assert "#FFF6D6" in text and "В норме" in text
    assert len(app.get("download_button")) == 1
    assert app.session_state["interpretation"] == before


@pytest.mark.parametrize("kind,page", [
    ("finished", "latest_pretest.py"),
    ("neuromatics", "neuromatics_pretest.py"),
])
def test_loading_local_page_and_missing_upload_never_dispatch_live_calls(kind, page, monkeypatch):
    monkeypatch.delenv("AIPM_LATEST_RESULT_JSON", raising=False)
    monkeypatch.setattr(latest_pipeline, "run_latest_analysis", lambda **_: pytest.fail("Unexpected analysis"))
    app = AppTest.from_file(str(ROOT / "app_pages" / page)).run()
    assert not app.exception
    assert [button.label for button in app.button] == ["Проанализировать ролик"]
    assert app.button[0].proto.type == "primary"
    app.button(key=f"latest_{kind}_live").click().run()
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
    assert len(app.tabs) == 3
    assert any('>+1%<' in item.value for item in app.markdown)
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
    assert len(app.button) == 1


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
def test_live_action_passes_material_kind_and_uses_selected_repeat_count(kind, page, monkeypatch):
    from io import BytesIO
    import streamlit as st

    monkeypatch.delenv("AIPM_LATEST_RESULT_JSON", raising=False)
    monkeypatch.delenv("AIPM_NEUROMATICS_RESULT_JSON", raising=False)
    uploaded = BytesIO(b"placeholder-video")
    uploaded.name = "clip.mp4"
    monkeypatch.setattr(st, "file_uploader", lambda *_, **__: uploaded)
    calls = []
    reviews = []
    from aipm3 import interpretation_checker

    def review(draft, **kwargs):
        reviews.append(draft)
        return {"status": "unavailable"}

    monkeypatch.setattr(interpretation_checker, "review_draft", review)

    def explain(*_, **__):
        content = interpretation()
        content.update(material_kind=kind, manager_semantic={
            "profiles": {task: [] for task in 'nmr'}, "fingerprint": "test-draft"})
        return content

    def analyze(**kwargs):
        calls.append(kwargs)
        return {"material_kind": kind, "scoring_version": latest_runtime.scoring_version_for(kind),
                "scores": {"Q": .04, "OPM": .2}}

    monkeypatch.setattr(latest_pipeline, "run_latest_analysis", analyze)
    monkeypatch.setattr(latest_interpretation, "build_latest_interpretation", explain)
    app = AppTest.from_file(str(ROOT / "app_pages" / page))
    app.secrets["VSELLM_API_KEY"] = "test-key"
    app.run(timeout=30)
    assert not app.exception
    assert not calls
    assert not reviews
    assert any("Анализ и проверка наблюдений" in caption.value for caption in app.caption)
    if kind == "neuromatics":
        assert any("10 раз; итоговые оценки усредняются" in caption.value for caption in app.caption)
    vertical = app.selectbox(key=f"latest_{kind}_vertical")
    selected_vertical = vertical.options[-1]
    vertical.select(selected_vertical).run(timeout=30)
    assert not app.exception
    assert not calls
    # Replacing a file or rerunning the page must not dispatch the pipeline.
    uploaded = BytesIO(b"replacement-video")
    uploaded.name = "replacement.mp4"
    app.run(timeout=30)
    assert not app.exception
    assert not calls
    app.button(key=f"latest_{kind}_live").click().run(timeout=30)
    assert not app.exception
    assert len(calls) == 1
    assert calls[0]["material_kind"] == kind
    assert calls[0]["repeat_count"] == (10 if kind == "neuromatics" else 3)
    assert calls[0]["allow_live"] is True
    assert calls[0]["api_key"] == "test-key"
    assert len(reviews) == 1
    assert app.session_state[f"latest_{kind}_result"]["source_name"] == "replacement.mp4"
    assert len(app.tabs) == 3
    app.run(timeout=30)
    assert not app.exception
    assert len(calls) == 1
    app.selectbox(key=f"latest_{kind}_vertical").select_index(0).run(timeout=30)
    assert not app.exception
    assert len(calls) == 1
    uploaded = None
    app.run(timeout=30)
    assert not app.exception
    assert len(calls) == 1
    assert len(reviews) == 1


@pytest.mark.parametrize("status,label", [("matched", "Попали в UVP"), ("absent", "Не попали в UVP")])
@pytest.mark.parametrize('kind', ['finished', 'neuromatics'])
def test_both_results_show_directions_idea_and_uvp_without_weights(status, label, kind):
    from aipm3.latest_uvp import target_for_vertical
    source = (ROOT / "app_pages/latest_pretest.py").read_text().split('\nif __name__ in {')[0]
    app = AppTest.from_string(source + '\nshow_latest_result(st.session_state["result"], st.session_state["interpretation"])\n')
    result = {"scores": {"Q": .04, "OPM": .2}, "source_sha": "a" * 64,
              "material_kind": kind, "scoring_version": latest_runtime.scoring_version_for(kind),
              "main_idea": "На Авито можно найти надёжного мастера.",
              "vertical_uvp": {"source_sha": "a" * 64, "target": target_for_vertical("Services"),
                               "status": status, "total": 30, "answers": [],
                               "counts": {"matched": 30 if status == "matched" else 0,
                                          "partial": 0, "absent": 30 if status == "absent" else 0,
                                          "contradicted": 0}}}
    content = interpretation()
    content["material_kind"] = kind
    content["details"]["r"]["drivers"][0]["contribution"] = .05
    # Even an externally supplied explanation cannot expose a fixed input.
    fixed = deepcopy(content["details"]["n"]["drivers"][0])
    fixed.update(feature="pack_shot_duration_seconds", label="Финальный кадр с брендом")
    content["details"]["n"]["drivers"].append(fixed)
    before = deepcopy(result)
    app.session_state["result"] = result
    app.session_state["interpretation"] = content
    app.run(timeout=30)
    assert not app.exception
    text = ' '.join(x.value for x in app.markdown) + ' '.join(x.value for x in app.caption)
    assert "+ В плюс в этом ролике" in text and "− В минус в этом ролике" in text
    if kind == 'neuromatics':
        assert "Финальный кадр с брендом" not in text
    assert "На Авито можно найти надёжного мастера." in text
    assert any(x.value == label for x in [*app.success, *app.error])
    assert not app.get("plotly_chart")
    assert "Вес среди" not in text and "упорядочены по весу" not in text
    assert not any("вес" in x.label.lower() or "вклад" in x.label.lower() for x in app.expander)
    assert app.session_state["result"] == before


@pytest.mark.parametrize("kind,page", [("finished", "latest_pretest.py"), ("neuromatics", "neuromatics_pretest.py")])
def test_uvp_runs_with_both_analyses_and_retains_scores(kind, page, monkeypatch):
    from io import BytesIO
    import streamlit as st
    from aipm3 import latest_uvp

    monkeypatch.delenv("AIPM_LATEST_RESULT_JSON", raising=False)
    uploaded = BytesIO(b"synthetic-video")
    uploaded.name = "clip.mp4"
    monkeypatch.setattr(st, "file_uploader", lambda *_, **__: uploaded)
    calls = []
    def analyze(**kwargs):
        calls.append("video")
        return {"material_kind": kind, "scoring_version": latest_runtime.scoring_version_for(kind),
                "vertical": kwargs["vertical"], "source_sha": "a" * 64,
                "scores": {"Q": .04, "OPM": .2},
                "evidence": {"fresh": [{"main_claim": "На Авито можно найти надёжного мастера."}]}}
    def uvp(result, target, *_):
        calls.append("uvp")
        assert target == latest_uvp.target_for_vertical("Services")
        assert result["scores"] == {"Q": .04, "OPM": .2}
        assert result["main_idea"] == "На Авито можно найти надёжного мастера."
        return dict(result, vertical_uvp={"source_sha": "a" * 64, "target": target, "status": "insufficient"})
    monkeypatch.setattr(latest_pipeline, "run_latest_analysis", analyze)
    monkeypatch.setattr(latest_uvp, "with_uvp", uvp)
    def explain(*_, **__):
        value = interpretation()
        value["material_kind"] = kind
        return value
    monkeypatch.setattr(latest_interpretation, "build_latest_interpretation", explain)
    app = AppTest.from_file(str(ROOT / "app_pages" / page))
    app.secrets["VSELLM_API_KEY"] = "test-key"
    app.run(timeout=30)
    app.selectbox(key=f"latest_{kind}_vertical").select("Услуги").run(timeout=30)
    assert not calls
    app.button(key=f"latest_{kind}_live").click().run(timeout=30)
    assert not app.exception
    assert calls == ["video", "uvp"]
    assert app.session_state[f"latest_{kind}_result"]["scores"] == {"Q": .04, "OPM": .2}
    app.run(timeout=30)
    assert not app.exception
    assert calls == ["video", "uvp"]
