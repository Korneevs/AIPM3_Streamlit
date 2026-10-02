import os

import streamlit as st


LATEST_PRETEST_ENABLED = os.environ.get("AIPM_ENABLE_LATEST_PRETEST", "1") == "1"


st.set_page_config(page_title="AI-Pretest MesSage · AIPM 3.0", page_icon="🎬", layout="wide")


def require_login() -> None:
    if st.session_state.get("mm_user"):
        return

    st.title("🎬 AI-Pretest MesSage")
    st.subheader("Вход")
    st.caption("Введите ваш логин Mattermost, чтобы продолжить.")
    with st.form("login_form"):
        login = st.text_input("Логин Mattermost", placeholder="например, i.ivanov")
        submitted = st.form_submit_button("Войти", type="primary")
    if submitted:
        clean = login.strip().lstrip("@")
        if clean:
            st.session_state["mm_user"] = clean
            st.rerun()
        st.error("Введите логин.")
    st.stop()


def home() -> None:
    st.title("🎬 AI-Pretest MesSage · AIPM 3.0")
    st.caption("Предтест рекламных креативов по заметности, считываемости и запоминаемости")
    st.markdown(
        """
        Загрузите готовый ролик в разделе **Видео-претест**. AIPM 3.0 объединяет
        три оценки: заметность, считываемость главной идеи и запоминаемость.

        Приложение показывает, **что модель учла в плюс и в минус**:
        наблюдения по ролику, их вклад в оценку и варианты для проверки.
        Неподтверждённые причины отмечены отдельно.
        """
    )
    if LATEST_PRETEST_ENABLED:
        st.markdown("Проверка попадания в UVP доступна в разделе **Предыдущая версия**.")


require_login()

with st.sidebar:
    st.caption(f"👤 Вы вошли как **{st.session_state['mm_user']}**")
    if st.button("Выйти", use_container_width=True):
        del st.session_state["mm_user"]
        st.rerun()

pages = [
    st.Page(home, title="Главная", icon="🏠", default=True),
]
if LATEST_PRETEST_ENABLED:
    pages.extend([
        st.Page("app_pages/latest_pretest.py", title="Видео-претест", icon="🎬", url_path="video_pretest"),
        st.Page("app_pages/video_pretest.py", title="Предыдущая версия", icon="📂", url_path="previous_pretest"),
    ])
else:
    pages.append(st.Page("app_pages/video_pretest.py", title="Видео-претест", icon="🎬", url_path="video_pretest"))
st.navigation(pages).run()
