"""Offline checks for upload, queue, and result isolation across UVP selections."""
from contextlib import contextmanager
from copy import deepcopy
from io import BytesIO

import streamlit as st
from streamlit.testing.v1 import AppTest

from app_pages import uvp_pretest as page


def test_upload_only_runs_on_submit_and_clears_stale_results(monkeypatch):
    upload = BytesIO(b"example clip")
    upload.name = "clip.mp4"
    calls = []
    source_paths = []
    monkeypatch.setattr(st, "file_uploader", lambda *a, **kw: upload)
    monkeypatch.setattr(page, "api_key", lambda: "offline-test")

    @contextmanager
    def slot(*, wait, on_queue):
        assert wait
        on_queue(2)
        on_queue(0)
        yield

    def media(path, **_):
        assert path.read_bytes() == b"example clip"
        source_paths.append(path)
        return {}

    def score(**kw):
        calls.append(kw["vertical"])
        kw["on_progress"]("Проверяем сцены")
        return {
            "main_idea": "Через Авито нашли мастера.",
            "images": [{"id": "service", "label": "Выбор услуг", "group": "Ассортимент", "score": 3,
                        "review": False, "clarity": 3, "limitation": "",
                        "explanation": "Через один сервис решают разные задачи.",
                        "evidence": [{"observation": "Герой вызывает мастера.", "time": "0:03"}]}],
        }

    monkeypatch.setattr(page, "analysis_slot", slot)
    monkeypatch.setattr(page, "prepare_evidence", media)
    monkeypatch.setattr(page, "score_video", score)
    app = AppTest.from_string("from app_pages.uvp_pretest import main; main()").run()
    assert not app.exception and app.button[0].disabled
    app.selectbox[0].select("Услуги").run()
    assert not calls and not app.button[0].disabled
    app.button[0].click().run()
    assert not app.exception and calls == ["Услуги"]
    assert "Основная идея по UVP-разбору" in [item.value for item in app.subheader]
    assert not app.warning and not source_paths[0].exists()
    app.run()
    assert calls == ["Услуги"]
    app.selectbox[0].select("Авто").run()
    assert calls == ["Услуги"] and not app.subheader
    app.selectbox[0].select("Услуги").run()
    assert "Основная идея по UVP-разбору" in [item.value for item in app.subheader]
    upload.seek(0)
    upload.write(b"different clip")
    app.run()
    assert calls == ["Услуги"] and not app.subheader


def test_summary_precedes_details_and_review_flags_do_not_change_visible_scores(monkeypatch):
    result = {
        "main_idea": "Найти мастера через Авито.",
        "images": [
            {"id": "general", "label": "Хорошие истории", "group": "Общие имиджи", "score": 3,
             "review": True, "clarity": 3, "limitation": "",
             "explanation": "Семья довольна ремонтом.",
             "evidence": [{"observation": "Семья в обновлённой комнате.", "time": "0:03"}]},
            {"id": "trust", "label": "Можно доверять", "group": "Доверие", "score": 2,
             "review": True, "clarity": 2, "limitation": "Договор в кадре не показан.",
             "explanation": "Показан положительный отзыв.",
             "evidence": [{"observation": "Клиент рекомендует мастера.", "time": "0:05"}]},
            {"id": "negative", "label": "Удобно пользоваться", "group": "Удобство и скорость",
             "score": -1, "review": False, "clarity": 2, "limitation": "",
             "explanation": "Герой долго не может оформить заявку.", "evidence": []},
        ],
    }
    original = deepcopy(result)
    html, figures = [], []
    monkeypatch.setattr(st, "html", lambda body, **kwargs: html.append(body))
    monkeypatch.setattr(st, "plotly_chart", lambda figure, **kwargs: figures.append(figure))
    app = AppTest.from_string(
        "import streamlit as st\nfrom app_pages.uvp_pretest import show_result\n"
        "show_result(st.session_state['fixture_result'])"
    )
    app.session_state["fixture_result"] = result
    app.run()
    assert not app.exception
    assert [item.value for item in app.subheader][:4] == [
        "Основная идея по UVP-разбору", "Основные UVP", "Основные имиджи", "Как ролик работает на имиджи"
    ]
    visible = "\n".join(html + [item.value for item in app.caption] + [item.value for item in app.markdown])
    assert "<b>Доверие</b> — Умеренно выражено · +2" in visible
    assert "Договор в кадре не показан." in visible
    assert "Клиент рекомендует мастера." in visible
    assert "Требует проверки" not in visible and "требует уточнения" not in visible.lower()
    trace = figures[0].data[0]
    assert list(trace.x) == [3, 2, -1]
    assert trace.marker.symbol == "circle"
    assert all("проверк" not in text.lower() for text in trace.customdata)
    assert result == original and result["images"][0]["review"] is True
