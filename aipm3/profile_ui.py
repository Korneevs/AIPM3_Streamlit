"""Business topics, plain-language observations, and within-outcome priorities."""
from __future__ import annotations

from html import escape

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .marketing_profile import build_marketing_profile


DIRECTIONS = {
    "up": ("В сумме помогает", "#137547"),
    "down": ("В сумме снижает", "#B42332"),
    "balanced": ("Плюсы и минусы уравновешены", "#667085"),
}


def share_text(value):
    return "<1%" if 0 < value < 1 else f"{value:.0f}%"


def topic_figure(outcomes):
    visible = [row for row in outcomes if row["available"]]
    figure = go.Figure()
    for field, sign, color, name in [
        ("negative_share", -1, "#DA7165", "Снижает прогноз"),
        ("positive_share", 1, "#16A370", "Помогает прогнозу"),
    ]:
        figure.add_trace(go.Bar(
            x=[sign * row[field] for row in visible],
            y=[row["label"] for row in visible], orientation="h",
            name=name, marker_color=color, width=.42,
            customdata=[[row[field], row["importance"]] for row in visible],
            hovertemplate="%{y}<br>" + name + ": %{customdata[0]:.1f}% вкладов"
                          "<br>Важность всей темы: %{customdata[1]:.1f}%<extra></extra>",
        ))
    figure.update_layout(
        height=65 + 36 * len(visible), barmode="relative", showlegend=False,
        margin=dict(l=0, r=22, t=5, b=25),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(size=12),
        xaxis=dict(range=[-100, 100], tickvals=[-100, -50, 0, 50, 100],
                   ticktext=["100%", "50%", "0", "50%", "100%"], fixedrange=True,
                   zeroline=True, zerolinecolor="#A9B0BC", showgrid=False,
                   ticklabeloverflow="allow"),
        yaxis=dict(autorange="reversed", fixedrange=True, automargin=True),
    )
    return figure


def show_feature_profile(result):
    profile = build_marketing_profile(result)
    st.subheader("Что помогает ролику, а что мешает")
    st.caption(
        "Разбор по пяти темам: от предложения до просмотра фрагментов. "
        "Основан на автоматической оценке ролика. "
        "Считываемость — насколько понятно основное предложение."
    )

    with st.container(border=True):
        st.markdown("#### С чего начать разбор")
        st.caption("Для каждой оценки — тема с самым большим суммарным отрицательным вкладом. "
                   "Это порядок проверки, а не обещание роста после правки.")
        for column, priority in zip(st.columns(3), profile["priorities"]):
            with column:
                st.markdown("**" + priority["label"] + "**")
                if not priority["available"]:
                    st.caption("Недостаточно данных для выбора приоритета.")
                elif priority["topics"]:
                    for topic in priority["topics"]:
                        st.markdown(f'[{topic["title"]}](#profile-{topic["id"]})')
                    if len(priority["topics"]) > 1:
                        st.caption("Одинаковый вклад — общий приоритет.")
                else:
                    st.caption("Нет тем с отрицательным суммарным вкладом.")

    st.markdown("**Темы весят по-разному.** Важность — вес темы в расчёте конкретной оценки "
                "для этого ролика. Большой вес бывает и у сильной стороны, и у слабой. "
                "В подробностях показаны веса отдельных свойств: они тоже различаются.")
    st.caption("Зелёная часть помогает прогнозу, красная — снижает. Обе части вместе составляют важность темы. "
               "Сравнивайте проценты только внутри одной оценки. Они не показывают долю зрителей или рост после правки.")

    for topic in profile["topics"]:
        with st.container(border=True):
            st.markdown(f'<div id="profile-{topic["id"]}" style="scroll-margin-top:80px;"></div>', unsafe_allow_html=True)
            description, effects = st.columns([1, 1.15], gap="large")
            with description:
                st.markdown("#### " + topic["title"])
                st.markdown(topic["question"])
                st.caption(topic["checks"])
            with effects:
                for row in topic["outcomes"]:
                    if not row["available"]:
                        st.markdown(f'**{row["label"]}** · недостаточно данных')
                        continue
                    status, color = DIRECTIONS[row["direction"]]
                    if row["importance"] == 0:
                        status = "Нет вклада в эту оценку"
                    st.markdown(
                        f'<div style="margin:5px 0 8px;line-height:1.5;font-size:14px;">'
                        f'<strong>{escape(row["label"])}</strong> · '
                        f'<span style="color:{color};font-weight:600;">{status}</span>'
                        f'<span> · Важность: '
                        f'<strong>{escape(share_text(row["importance"]))}</strong></span></div>',
                        unsafe_allow_html=True,
                    )
                if any(row["available"] for row in topic["outcomes"]):
                    st.plotly_chart(topic_figure(topic["outcomes"]), use_container_width=True,
                                    config={"displayModeBar": False}, key="topic_" + topic["id"])
            with st.expander("Что это значит и что найдено в ролике"):
                st.caption("Ниже — результаты автоматического разбора. «Есть» или «не обнаружено» "
                           "описывает ролик; это само по себе не хорошо и не плохо.")
                if not topic["observations"]:
                    st.write("Нет совместимых данных для подробного разбора.")
                for label, definition, value in topic["observations"]:
                    st.markdown(f"**{label}: {value}**")
                    st.caption(definition)
                if topic.get("combined_checks"):
                    st.markdown("**Что оценивается в сочетании**")
                    st.write(topic["combined_checks"])
                st.markdown("**Вклад отдельных свойств в этом ролике**")
                st.caption("Сверху — самые весомые свойства внутри темы. Проценты относятся ко всей "
                           "соответствующей оценке, а не только к этой теме. Отсутствие приёма тоже может "
                           "помогать прогнозу — наличие всех приёмов не является целью.")
                for row in topic["outcomes"]:
                    if row["available"]:
                        st.markdown("**" + row["label"] + "**")
                        details = [{"Что учитывается": driver["label"],
                                    "Направление": {"up": "Помогает", "down": "Снижает", "balanced": "Нет вклада"}[driver["direction"]],
                                    "Важность": share_text(driver["importance"])} for driver in row["drivers"]]
                        st.table(pd.DataFrame(details).set_index("Что учитывается"))
                st.markdown("**Что проверить при просмотре**")
                st.write(topic["look_at"])

    with st.expander("Почему веса разные и как выбран приоритет"):
        st.write("Одинаковых постоянных весов у свойств ролика нет. Программа оценивает их "
                 "в сочетании: например, эффект музыки зависит от речи, а эффект героя — от сюжета. "
                 "Здесь показано, как свойства объясняют прогноз именно для загруженного ролика.")
        st.write("Внутри каждой из трёх оценок за 100% взята суммарная сила всех положительных "
                 "и отрицательных вкладов свойств ролика. Важность темы — её доля в этой сумме. "
                 "Для каждой оценки важность всех тем в сумме составляет 100%, кроме случая, "
                 "когда все вклады нулевые. Округление может дать небольшое отклонение от 100%.")
        st.write("У одной темы могут быть и плюсы, и минусы. Важность учитывает обе стороны, "
                 "а подпись «в сумме помогает / снижает» — их разницу. Первой для проверки "
                 "выбрана тема с самым большим оставшимся минусом. Большой вес сам по себе "
                 "не означает, что тему нужно исправлять.")
        st.write("Одна тема может влиять на несколько оценок по-разному. Поэтому её вклады "
                 "показаны отдельно: мы не усредняем заметность, запоминаемость и считываемость. "
                 "Фиксированный контекст бренда учитывается в общей запоминаемости, но не входит "
                 "в этот профиль свойств ролика.")
        st.write("При изменении ролика изменятся и сочетания свойств. Поэтому вклад не равен "
                 "ожидаемому приросту от конкретной правки. Новую версию нужно оценить заново. "
                 "Даже положительный вклад не гарантирует высокий общий результат.")
        st.caption("Для аналитиков: используются сохранённые SHAP-вклады. Для заметности — "
                   "перевес высокого класса над низким; для запоминаемости и считываемости — "
                   "непрерывный прогноз. Вклады объединены по темам без переобучения моделей. "
                   "Это объяснение прогноза, не разложение итогового AIPM 3.0 и не измеренный причинный эффект.")
