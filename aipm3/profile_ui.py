"""Numerical, higher-is-better group scales without generated commentary."""
from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from .feature_profile import GROUPS, build_profile


def group_figure(rows):
    figure = go.Figure()
    positions = list(range(len(rows)))
    figure.add_trace(go.Bar(
        x=[100] * len(rows), y=positions, orientation="h",
        marker_color="#EDF0F4", hoverinfo="skip", width=.16,
    ))
    visible = [(pos, row) for pos, row in enumerate(rows) if row["score"] is not None]
    figure.add_trace(go.Bar(
        x=[row["score"] for _, row in visible], y=[pos for pos, _ in visible],
        orientation="h", width=.16,
        marker_color=["#16A370" if row["display_score"] > 50 else
                      "#DA7165" if row["display_score"] < 50 else "#8792A5"
                      for _, row in visible],
        customdata=[[row["label"], row["display_score"]] for _, row in visible],
        hovertemplate="%{customdata[0]}<br>%{customdata[1]} / 100<extra></extra>",
    ))
    figure.add_shape(type="line", x0=50, x1=50, y0=-.15, y1=len(rows) - .8,
                     line=dict(color="#A9B0BC", dash="dot", width=1), layer="above")
    for pos, row in enumerate(rows):
        label = row["label"].replace("Полнота и конкретность предложения",
                                      "Полнота и конкретность<br>предложения").replace(
            "Устойчивость при неполном просмотре", "Устойчивость при неполном<br>просмотре")
        figure.add_annotation(x=0, y=pos - .30, text=label, xanchor="left",
                              yanchor="bottom", showarrow=False, align="left",
                              font=dict(size=14, color="#343745"))
        figure.add_annotation(
            x=100, y=pos + .24, xanchor="right", showarrow=False,
            text=f'<b>{row["display_score"]} / 100</b>' if row["score"] is not None else "Нет данных",
            font=dict(size=15, color="#343745"),
        )
    figure.update_layout(
        height=390, barmode="overlay", margin=dict(l=10, r=10, t=15, b=30),
        showlegend=False, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(range=[0, 100], tickvals=[0, 50, 100], fixedrange=True,
                   ticklabeloverflow="allow", showgrid=False, zeroline=False,
                   tickfont=dict(size=12, color="#818591")),
        yaxis=dict(range=[3.65, -.90], visible=False, fixedrange=True),
    )
    return figure


def show_feature_profile(result):
    profile = build_profile(result)
    st.subheader("Профиль ролика по группам признаков")
    st.caption("Оценка каждой группы: 0–100. Больше — лучше для прогноза модели. "
               "50 — нейтральный вклад; ниже — слабая сторона, выше — сильная.")
    for column, (component, (title, _)) in zip(st.columns(3), GROUPS.items()):
        with column, st.container(border=True):
            st.markdown("**" + title + "**")
            rows = [row for row in profile["groups"] if row["component"] == component]
            st.plotly_chart(group_figure(rows), use_container_width=True,
                            config={"displayModeBar": False}, key="profile_" + component)
    validation = profile["validation"]
    with st.expander("Как считаются оценки групп"):
        st.write("В каждой группе объединены прежние признаки. Балл рассчитывается по тому, "
                 "как их сочетание поддерживает прогноз соответствующей модели. "
                 "Отдельная нейросеть не пишет объяснение и не выставляет эти баллы.")
        st.write("50 означает нулевой суммарный вклад группы относительно базового прогноза. "
                 "Положительный вклад переводится в 50–100, отрицательный — в 0–50. "
                 "Масштаб зафиксирован по историческим данным и одинаков для всех групп "
                 "внутри одной модели. Он не подстраивается под загруженный ролик.")
        st.write("Это модельная оценка в контексте всего ролика, не процент зрителей и не "
                 "доказательство причинного влияния приёма. Например, наличие юмора само по себе "
                 "не гарантирует высокий балл. Группы могут взаимодействовать; их оценки нельзя "
                 "складывать или усреднять для получения трёх основных метрик.")
        st.write("Под капотом суммируются уже рассчитанные вклады признаков (SHAP), "
                 "затем применяется фиксированная монотонная шкала. Для заметности объясняется "
                 "перевес высокого класса над низким, для двух других моделей — их непрерывный прогноз. "
                 "Веса, разметка и итоговые показатели моделей не меняются.")
        if validation.get("completed"):
            st.write(validation["summary"])
        else:
            st.write("Межзапусковая проверка новых шкал ещё не завершена. "
                     "Одинаковые сохранённые признаки всегда дают одинаковые баллы; "
                     "новая разметка Gemini может отличаться.")
