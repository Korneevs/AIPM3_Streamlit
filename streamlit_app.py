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
    st.caption("Предтест рекламных креативов по заметности, запоминаемости и считываемости")
    st.markdown(
        """
        Загрузите готовый ролик в разделе **Видео-претест**. AIPM 3.0 объединяет
        три независимых сигнала: AIPM 1.0, AIPM 2.0 и Message Delivery.

        Приложение показывает, **что в ролике работает** и
        **что ограничивает результат**, сначала по трём компонентам, затем по
        семи темам: предложение, главная мысль, бренд, музыка, речь, сюжет и просмотр фрагментов.
        Для каждой темы показаны вывод по загруженному ролику, вес в оценках и подробности
        о том, что найдено.

        ---
        Ключи и замороженные модели хранятся в Streamlit Secrets и не находятся в коде.
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
    st.Page("app_pages/video_pretest.py", title="Видео-претест", icon="🎬"),
]
st.navigation(pages).run()
