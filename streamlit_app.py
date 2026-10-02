import streamlit as st


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
        Выберите раздел **AIPM3.0 (для готовых)** или **AIPM3.0 (для нейроматиков)**
        и загрузите ролик. AIPM 3.0 объединяет
        три оценки: заметность, считываемость главной идеи и запоминаемость.

        Приложение показывает, **что модель учла в плюс и в минус**:
        наблюдения по ролику, их вклад в оценку и варианты для проверки.
        Неподтверждённые причины отмечены отдельно.
        """
    )


require_login()

with st.sidebar:
    st.caption(f"👤 Вы вошли как **{st.session_state['mm_user']}**")
    if st.button("Выйти", use_container_width=True):
        del st.session_state["mm_user"]
        st.rerun()

pages = [
    st.Page(home, title="Главная", icon="🏠", default=True),
]
pages.extend([
    st.Page("app_pages/latest_pretest.py", title="AIPM3.0 (для готовых)", icon="🎬", url_path="video_pretest"),
    st.Page("app_pages/neuromatics_pretest.py", title="AIPM3.0 (для нейроматиков)", icon="📝", url_path="neuromatics_pretest"),
])
st.navigation(pages).run()
