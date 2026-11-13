"""Offline checks for upload, queue, and result isolation across UVP selections."""
from contextlib import contextmanager
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
    assert "Основная идея ролика" in [item.value for item in app.subheader]
    assert not app.warning and not source_paths[0].exists()
    app.run()
    assert calls == ["Услуги"]
    app.selectbox[0].select("Авто").run()
    assert calls == ["Услуги"] and not app.subheader
    app.selectbox[0].select("Услуги").run()
    assert "Основная идея ролика" in [item.value for item in app.subheader]
    upload.seek(0)
    upload.write(b"different clip")
    app.run()
    assert calls == ["Услуги"] and not app.subheader
