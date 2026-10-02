"""Exercise both material routes and keep all navigation offline."""
from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest
from streamlit.util import calc_md5
from streamlit.runtime.pages_manager import PagesManager
from streamlit.runtime.scriptrunner.script_cache import ScriptCache

from aipm3 import latest_pipeline


ROOT = Path(__file__).resolve().parents[1]
ROUTES = {
    "video_pretest": ("AIPM3.0 (для готовых)", "latest_pretest.py"),
    "neuromatics_pretest": ("AIPM3.0 (для нейроматиков)", "neuromatics_pretest.py"),
}


@pytest.fixture
def offline_navigation(monkeypatch):
    monkeypatch.delenv("AIPM_LATEST_RESULT_JSON", raising=False)
    monkeypatch.delenv("AIPM_NEUROMATICS_RESULT_JSON", raising=False)
    monkeypatch.delenv("AIPM_ENABLE_LATEST_PRETEST", raising=False)

    def forbidden(*args, **kwargs):
        pytest.fail("Opening or navigating the app must not run inference")

    monkeypatch.setattr(latest_pipeline, "run_latest_analysis", forbidden)
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
        app._page_hash = calc_md5(path)
    return app.run()


@pytest.mark.parametrize("setting", [None, "1", "0"])
@pytest.mark.parametrize("path", ["video_pretest", "neuromatics_pretest"])
def test_only_current_material_routes_even_with_old_environment(setting, path, monkeypatch, offline_navigation):
    if setting is not None:
        monkeypatch.setenv("AIPM_ENABLE_LATEST_PRETEST", setting)
    app = open_url(path)
    assert not app.exception
    assert offline_navigation == ROUTES
    assert app.title[0].value == ROUTES[path][0]
    assert app.radio[0].label == "Источник результата"
    assert "Запустить AI-анализ" in [button.label for button in app.button]
    assert not any("UVP" in caption.value for caption in app.caption)


def test_old_url_cannot_open_legacy_page(offline_navigation):
    app = open_url("previous_pretest")
    assert not app.exception
    assert offline_navigation == ROUTES
    assert app.title[0].value == "🎬 AI-Pretest MesSage · AIPM 3.0"
    assert not any(button.label == "Начать анализ" for button in app.button)


def test_home_describes_both_sections(offline_navigation):
    app = open_url()
    assert not app.exception
    text = " ".join(item.value for item in app.markdown)
    assert "заметность, считываемость главной идеи и запоминаемость" in text
    assert "что модель учла в плюс и в минус" in text
    assert "AIPM3.0 (для готовых)" in text
    assert "AIPM3.0 (для нейроматиков)" in text
    assert "Предыдущая версия" not in text


def test_navigation_keeps_material_widgets_separate(offline_navigation):
    app = open_url("video_pretest")
    app.selectbox[0].select("Работа").run()
    finished_upload_id = app.get("file_uploader")[0].proto.id
    assert app.selectbox[0].value == "Работа"
    app._page_hash = calc_md5("neuromatics_pretest")
    app.run()
    assert not app.exception
    assert app.title[0].value == ROUTES["neuromatics_pretest"][0]
    assert app.selectbox[0].value == "Товары"
    assert app.get("file_uploader")[0].proto.id != finished_upload_id
    assert app.radio[0].key == "latest_neuromatics_source"
    app.radio[0].set_value("Открыть сохранённый результат").run()
    assert app.get("file_uploader")[0].proto.id.endswith("latest_neuromatics_saved")
    app._page_hash = calc_md5("video_pretest")
    app.run()
    assert not app.exception
    assert app.radio[0].value == "Загрузить ролик"
    assert app.get("file_uploader")[0].proto.id.endswith("latest_finished_video")
