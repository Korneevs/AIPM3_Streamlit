"""Archive-style cards and feature tabs for the current model's explanations.

No legacy score conversion or legacy SHAP is used. Attribution weights use all
creative features of each current head, including unresolved contributions.
"""
from __future__ import annotations

from html import escape

import plotly.graph_objects as go
import streamlit as st

from .manager_report import LEVELS, PURPOSE, report_cards
from .latest_interpretation import LABELS


ORDER = ("n", "r", "m")
TITLES = {"n": "Заметность", "r": "Запоминаемость", "m": "Считываемость основной идеи"}


def show_summary(interpretation: dict) -> None:
    def render(card: dict, primary: bool = False) -> None:
        status, color, background = LEVELS[card["level"]]
        available = card.get("assessment_available", True)
        delta = round(float(card["index"]) - 100)
        value = ((f"{delta:+d}%" if delta else "0%").replace("-", "−") if available else "—")
        title = "Общая оценка · AIPM 3.0" if primary else TITLES[card["task"]]
        note = "к среднему" if available else "Нужна полная озвучка"
        if not available:
            color, background, status = "#667085", "#F5F6F8", "Оценка ограничена"
        st.markdown(
            f'<div style="background:{background};border:1px solid {color}33;'
            f'border-radius:12px;padding:{"24px" if primary else "18px 16px"};color:{color};'
            f'margin-bottom:{"16px" if primary else "0"};" '
            f'data-metric="{"overall" if primary else "component"}">'
            f'<div style="font-size:{18 if primary else 14}px;color:#343745;min-height:{24 if primary else 42}px;">{escape(title)}</div>'
            f'<div style="font-size:{60 if primary else 34}px;line-height:1.2;font-weight:700;margin-top:8px;">{value}</div>'
            f'<div style="font-size:13px;margin-top:4px;">{note}</div>'
            f'<div style="font-size:15px;font-weight:600;margin-top:16px;">{status}</div></div>',
            unsafe_allow_html=True,
        )

    if interpretation.get("overall"):
        render(interpretation["overall"], True)
    cards = {card["task"]: card for card in interpretation["cards"]}
    for column, task in zip(st.columns(3), ORDER):
        with column:
            render(cards[task])
    st.caption("0% - ориентир для нейроматиков, сопоставленный с готовыми роликами."
               if interpretation.get("material_kind") == "neuromatics" else
               "0% - средняя оценка роликов, с которыми сравниваем этот вариант.")


def profile_rows(interpretation: dict) -> dict[str, list[dict]]:
    """Every feature appears exactly once; business claims keep existing gates."""
    cards = {card["task"]: card for card in report_cards(interpretation)}
    result = {}
    for task in ORDER:
        card = cards[task]
        drivers = [d for d in interpretation["details"][task]["drivers"]
                   if d["feature"] != "brand_history"]
        mass = sum(abs(d["contribution"]) for d in drivers)
        claims = {d["feature"]: d for group in ("strengths", "limitations") for d in card[group]}
        reasons = {label: note["text"] for note in card["unassessed"] for label in note["features"]}
        rows = []
        for driver in drivers:
            claim = claims.get(driver["feature"])
            contribution = driver["contribution"]
            sign = "down" if contribution < 0 else "up" if contribution > 0 else "balanced"
            label = claim["label"] if claim else LABELS.get(driver["feature"], driver["label"])
            reason = reasons.get(label,
                "По сохранённым наблюдениям нельзя уверенно назвать эту деталь достоинством или недостатком этого ролика.")
            context = PURPOSE.get(driver["feature"], "")
            rows.append(dict(
                feature=driver["feature"], label=label,
                importance=100 * abs(contribution) / mass if mass else 0,
                direction=sign, claim=claim,
                finding=claim["takeaway"] if claim else (context + " " + reason).strip(),
                observation=claim.get("evidence", "") if claim else "",
                check=claim.get("check", "") if claim else "",
            ))
        rows.sort(key=lambda row: ({"down": 0, "up": 1, "balanced": 2}[row["direction"]], -row["importance"]))
        result[task] = rows
    return result


def feature_figure(row: dict) -> go.Figure:
    value = row["importance"] * (-1 if row["direction"] == "down" else 1)
    figure = go.Figure(go.Bar(
        x=[value], y=["Вклад в оценку"], orientation="h", width=.42,
        marker_color="#DA7165" if value < 0 else "#16A370",
        customdata=[row["importance"]],
        hovertemplate="Вес свойства: %{customdata:.1f}%<extra></extra>",
    ))
    figure.update_layout(
        height=100, showlegend=False, margin=dict(l=0, r=22, t=5, b=25),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(range=[-100, 100], tickvals=[-100, -50, 0, 50, 100],
                   ticktext=["100%", "50%", "0", "50%", "100%"], fixedrange=True,
                   zeroline=True, zerolinecolor="#A9B0BC", showgrid=False),
        yaxis=dict(showticklabels=False, fixedrange=True),
    )
    return figure


def show_profile(interpretation: dict) -> None:
    rows = profile_rows(interpretation)
    st.subheader("Профиль ролика")
    for tab, task in zip(st.tabs([TITLES[t] for t in ORDER]), ORDER):
        with tab:
            if task == "r" and interpretation.get("celebrity_present") is True:
                st.success("В конкретном ролике участие медийной персоны положительно влияет на оценку запоминаемости. Участие отмечено вами.")
            for row in rows[task]:
                with st.container(border=True):
                    description, effects = st.columns([1.5, 1], gap="large")
                    with description:
                        st.markdown("#### " + row["label"])
                        st.write(row["finding"])
                        if row["check"]:
                            st.markdown("**Вариант для проверки:** " + row["check"])
                    with effects:
                        status, color = {"up": ("Поддерживает оценку", "#137547"),
                                         "down": ("Снижает оценку", "#B42332"),
                                         "balanced": ("Нет вклада", "#667085")}[row["direction"]]
                        st.markdown(f'<div style="color:{color};font-weight:600;margin:12px 0 5px;">{status}</div>',
                                    unsafe_allow_html=True)
                        weight = "<1%" if 0 < row["importance"] < 1 else f'{row["importance"]:.0f}%'
                        st.write("Вес среди свойств ролика: **" + weight + "**")
                        st.plotly_chart(feature_figure(row), use_container_width=True,
                                        config={"displayModeBar": False},
                                        key=f'latest_profile_{task}_{row["feature"]}')
                    with st.expander("Что учтено и что означает вес"):
                        if row["observation"]:
                            st.write(row["observation"])
                        if not row["claim"]:
                            st.write("Направление справа показывает вклад в расчёт, но не подтверждённую причину успеха или недостаток ролика.")
                        st.caption("Вес показывает долю свойства в сумме вкладов в эту оценку. Это не ожидаемый рост после правки ролика.")
    with st.expander("Как рассчитан вклад"):
        st.write("В каждой вкладке сначала показаны свойства, снижающие оценку, затем поддерживающие. Внутри этих групп они упорядочены по весу.")
        st.write("Веса всех свойств внутри одной оценки составляют 100% до округления; если вкладов нет, веса равны нулю. Исторический ориентир бренда учтён отдельно. Веса разных оценок не складываются.")
        st.write("Вклад объясняет расчёт для конкретного ролика. Он не доказывает, что добавление или удаление приёма улучшит результат. Варианты правок нужно сравнивать.")
