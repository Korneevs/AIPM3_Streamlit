"""Exercise the published URLs through Streamlit navigation without inference."""
from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest
from streamlit.util import calc_md5
from streamlit.runtime.pages_manager import PagesManager
from streamlit.runtime.scriptrunner.script_cache import ScriptCache

from aipm3 import latest_pipeline, models, repeated_pipeline, vertical_uvp
import artifacts


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def offline_navigation(monkeypatch):
    monkeypatch.delenv("AIPM_LATEST_RESULT_JSON", raising=False)
    monkeypatch.delenv("AIPM_ENABLE_LATEST_PRETEST", raising=False)

    def forbidden(*args, **kwargs):
        pytest.fail("Opening or navigating the app must not run inference or UVP")

    monkeypatch.setattr(latest_pipeline, "run_latest_analysis", forbidden)
    monkeypatch.setattr(repeated_pipeline, "run_repeated_analysis", forbidden)
    monkeypatch.setattr(repeated_pipeline, "evaluate_repeated_uvp", forbidden)
    monkeypatch.setattr(vertical_uvp, "evaluate_uvp", forbidden)
    monkeypatch.setattr(artifacts, "artifact_path", lambda name: name)
    monkeypatch.setattr(models, "load_frozen_models", lambda *args: object())
    # AppTest creates PagesManager without its child-page script cache.
    script_cache = ScriptCache()
    monkeypatch.setattr(PagesManager, "get_page_script_byte_code",
                        lambda self, path: script_cache.get_bytecode(path))
    original_navigation = st.navigation
    routes = {}

    def capture_routes(pages, **kwargs):
        routes.clear()
        routes.update({page.url_path: (page.title, Path(page._page).name)
                       for page in pages if isinstance(page._page, Path)})
        return original_navigation(pages, **kwargs)

    monkeypatch.setattr(st, "navigation", capture_routes)
    return routes


def open_url(path=""):
    app = AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=30)
    app.session_state["mm_user"] = "routing-review"
    app.secrets["VSELLM_API_KEY"] = "never-used"
    if path:
        # AppTest.switch_page hashes filenames; st.navigation hashes URL paths.
        app._page_hash = calc_md5(path)
    return app.run()


@pytest.mark.parametrize("setting", [None, "1"])
def test_existing_pretest_url_opens_latest_by_default(setting, monkeypatch, offline_navigation):
    if setting is not None:
        monkeypatch.setenv("AIPM_ENABLE_LATEST_PRETEST", setting)
    app = open_url("video_pretest")
    assert not app.exception
    assert offline_navigation == {
        "video_pretest": ("Видео-претест", "latest_pretest.py"),
        "previous_pretest": ("Предыдущая версия", "video_pretest.py"),
    }
    assert app.title[0].value == "Видео-претест · AIPM 3.0"
    assert app.radio[0].label == "Источник результата"
    assert "Запустить AI-анализ" in [button.label for button in app.button]
    assert not any("UVP" in caption.value for caption in app.caption)


def test_previous_url_keeps_vertical_uvp_and_survives_rerun(offline_navigation):
    app = open_url("previous_pretest")
    assert not app.exception
    assert any(button.label == "Начать анализ" for button in app.button)
    assert not app.radio
    app.query_params["campaign"] = "publication-check"
    next(select for select in app.selectbox if select.label == "Вертикаль").select("Работа").run()
    assert not app.exception
    assert any(caption.value.startswith("Целевой UVP:") for caption in app.caption)
    assert app.query_params["campaign"] == ["publication-check"]
    assert any(button.label == "Начать анализ" for button in app.button)


def test_rollback_restores_previous_page_at_existing_url(monkeypatch, offline_navigation):
    monkeypatch.setenv("AIPM_ENABLE_LATEST_PRETEST", "0")
    app = open_url("video_pretest")
    assert not app.exception
    assert offline_navigation == {"video_pretest": ("Видео-претест", "video_pretest.py")}
    assert any(button.label == "Начать анализ" for button in app.button)
    assert any(select.label == "Селебрити в ролике" for select in app.selectbox)
    assert not app.radio


def test_home_describes_current_scores_and_places_uvp_in_previous_version(offline_navigation):
    app = open_url()
    assert not app.exception
    text = " ".join(item.value for item in app.markdown)
    assert "заметность, считываемость главной идеи и запоминаемость" in text
    assert "что модель учла в плюс и в минус" in text
    assert "Проверка попадания в UVP доступна в разделе **Предыдущая версия**" in text
    assert "После расчёта появится отдельная проверка" not in text
