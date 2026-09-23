"""Business topics with conclusions specific to the analyzed video."""
from __future__ import annotations

from html import escape

import plotly.graph_objects as go
import streamlit as st

from .marketing_profile import build_marketing_profile
from .celebrity_ui import show_celebrity


DIRECTIONS = {
    "up": ("Поддерживает", "#137547"),
    "down": ("Снижает", "#B42332"),
    "balanced": ("Нет вклада", "#667085"),
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
    components = [("aipm1", "Заметность"), ("aipm2", "Запоминаемость"),
                  ("message_delivery", "Считываемость")]
    for tab, (component, label) in zip(st.tabs([label for _, label in components]), components):
        with tab:
            if component == "aipm2":
                show_celebrity(result)
            manual_recall = component == "aipm2" and "celebrity_adjustment" in result
            weight_label = "Вес среди свойств ролика" if manual_recall else "Вес в этой оценке"
            for topic in [t for t in profile["topics"] if t["component"] == component]:
                row = topic["outcomes"][0]
                with st.container(border=True):
                    description, effects = st.columns([1.5, 1], gap="large")
                    with description:
                        st.markdown("#### " + topic["title"])
                        st.write(topic["finding"])
                        if any(d["feature"].startswith("eng__") for d in row["drivers"]):
                            st.caption("Здесь оценено сочетание свойств. Его вклад нельзя приписать одному из них.")
                        if row["available"] and row["direction"] == "down":
                            check = next((d["check"] for d in row["drivers"] if d["check"]), None)
                            if check:
                                st.markdown("**Вариант для проверки:** " + check)
                    with effects:
                        if not row["available"]:
                            st.write("Недостаточно данных для оценки вклада")
                        else:
                            status, color = DIRECTIONS[row["direction"]]
                            st.markdown(
                                f'<div style="color:{color};font-weight:600;margin:12px 0 5px;">'
                                f'{escape(status)} · {escape(label.lower())}</div>'
                                f'<div style="font-size:14px;">{weight_label}: '
                                f'<strong>{escape(share_text(row["importance"]))}</strong></div>',
                                unsafe_allow_html=True)
                            st.plotly_chart(topic_figure([row]), use_container_width=True,
                                            config={"displayModeBar": False}, key="topic_" + topic["id"])
                    with st.expander("Что учтено и что означает вес"):
                        st.write(topic["conclusion"])
                        for driver in row["drivers"]:
                            st.markdown("**" + driver["label"] + "**")
                            for name, definition, value in driver["observations"]:
                                st.write(name + ": " + value)
                                st.caption(definition)
                            if row["available"]:
                                st.caption("Вес свойства в оценке «" + label + "»: " + share_text(driver["importance"]))
                            if driver["feature"].startswith("eng__"):
                                st.caption("Это сочетание свойств: модель оценивает их вместе. Разделить этот вклад между отдельными приёмами без изменения модели нельзя.")
                        st.caption("Вес — доля этого свойства в силе всех вкладов в одну оценку данного ролика. "
                                   "Более высокий вес означает больший вклад в расчёт, а не гарантированный рост после правки.")
    with st.expander("Как рассчитан вклад"):
        st.write("Свойства ролика в каждой вкладке показаны по порядку: сначала снижающие её оценку, затем поддерживающие. "
                 "Внутри этих групп карточки упорядочены по весу. Разные направления и разные оценки показаны отдельно; "
                 "ни один вклад не скрыт и не учтён дважды.")
        st.write("Веса внутри одной оценки составляют 100% до округления; если все вклады нулевые, все веса равны нулю. "
                 "Фиксированный контекст бренда исключён. Веса разных оценок не складываются.")
        if "celebrity_adjustment" in result:
            st.write("Вес участия селебрити — его доля в итоговой запоминаемости. "
                     "Веса остальных свойств показывают их соотношение между собой и считаются отдельно.")
        st.caption("Это сохранённые SHAP-вклады: для заметности — перевес высокого класса над низким, "
                   "для запоминаемости и считываемости — непрерывный прогноз. Они объясняют расчёт модели, "
                   "а не доказывают причинное влияние. Варианты правок нужно проверять сравнением роликов.")
