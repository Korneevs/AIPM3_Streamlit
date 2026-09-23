"""Business topics with conclusions specific to the analyzed video."""
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
    st.subheader("Профиль ролика")

    for topic in profile["topics"]:
        with st.container(border=True):
            st.markdown(f'<div id="profile-{topic["id"]}" style="scroll-margin-top:80px;"></div>', unsafe_allow_html=True)
            description, effects = st.columns([1, 1.15], gap="large")
            with description:
                st.markdown("#### " + topic["title"])
                st.write(topic["finding"])
                st.markdown("**" + topic["conclusion"] + "**")
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
            with st.expander("Что учтено в оценке"):

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

    with st.expander("Как рассчитан вклад"):
        st.write("Важность темы — её доля в суммарной силе положительных и отрицательных "
                 "вкладов свойств этого ролика. Внутри одной оценки доли тем составляют 100% "
                 "до округления; если все вклады нулевые, все доли равны нулю. "
                 "Фиксированный контекст бренда в профиль не входит.")
        st.write("Зелёная часть поддерживает прогноз, красная — снижает. Подпись показывает "
                 "их суммарный результат. Веса отдельных свойств приведены в подробностях каждой темы. "
                 "Вклады заметности, запоминаемости и считываемости не складываются между собой.")
        st.caption("Используются сохранённые SHAP-вклады: для заметности — перевес высокого класса "
                   "над низким, для двух других оценок — непрерывный прогноз. Это объяснение модели "
                   "для данного ролика, а не доля зрителей и не прогноз роста после правки.")
