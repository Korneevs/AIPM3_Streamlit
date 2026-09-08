"""Manager-facing presentation of existing scores and saved diagnostic evidence."""
from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from .feature_profile import build_profile
from .manager_explanation import build_manager_explanation
from .models import AIPM3_Q33, AIPM3_Q67
from .profile_ui import show_feature_profile


def component_figure(components: list[dict]) -> go.Figure:
    values = [row["percentile"] for row in components]
    colors = ["#1677A8" if v >= 67 else "#B47A12" if v >= 33 else "#C64253" for v in values]
    figure = go.Figure(go.Bar(
        x=values, y=[row["label"] for row in components], orientation="h",
        marker_color=colors,
        text=[f"{v:.0f}" for v in values], textposition="outside", cliponaxis=False,
        hovertemplate="%{y}<br>Позиция в базе сравнения: %{x:.1f} из 100<extra></extra>",
    ))
    for value in [33, 67]:
        figure.add_vline(x=value, line_dash="dot", line_color="#B8C0CC")
    figure.update_layout(
        height=250, margin=dict(l=0, r=36, t=8, b=12), showlegend=False,
        xaxis=dict(range=[0, 107], tickvals=[0, 33, 67, 100],
                   title="Позиция в исторической базе · 0–100", fixedrange=True),
        yaxis=dict(autorange="reversed", title="", fixedrange=True, automargin=True),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    return figure


def show_main_message(evidence: dict) -> None:
    st.subheader("Как модель прочитала посыл")
    st.caption("Ответы модели после полного просмотра ролика. Это синтетические ответы, не опрос людей.")
    if evidence["status"] == "unavailable":
        st.info(evidence["summary"])
        for issue in evidence.get("issues", []):
            st.caption(issue)
        return
    if evidence["status"] == "partial":
        st.warning(evidence["summary"])
    else:
        st.write(evidence["summary"])
    for issue in evidence.get("issues", []):
        st.caption(issue)

    total = evidence["full_count"]
    rows = [
        {"Вариант прочтения": row["idea"], "Ответов": f"{row['count']} из {total}"}
        for row in evidence["clusters"]
    ]
    for label, key in [
        ("Назван только бренд", "brand_only_count"),
        ("Посыл не сформулирован", "no_idea_count"),
        ("Ответ не удалось надёжно классифицировать", "unclassified_count"),
    ]:
        if evidence[key]:
            rows.append({"Вариант прочтения": label, "Ответов": f"{evidence[key]} из {total}"})
    if rows:
        st.dataframe(rows, use_container_width=True, hide_index=True)
    st.caption("Эти ответы показывают возможные трактовки. Они не подтверждают совпадение с замыслом кампании: бриф здесь не задаётся.")
    with st.expander("Примеры ответов и ограничения"):
        for cluster in evidence["clusters"]:
            examples = cluster.get("examples", [])
            if examples:
                st.write(cluster["idea"])
                for example in examples:
                    st.text("«" + example + "»")
        st.write(evidence["caveat"])


def show_manager_result(result: dict, *, saved_profile: dict | None = None) -> tuple[dict, dict]:
    profile = saved_profile if saved_profile is not None else build_profile(result)
    explanation = build_manager_explanation({**result, "feature_profile": profile})
    total = result["aipm3"]
    st.subheader("Результат ролика")
    with st.container(border=True):
        st.metric("AIPM 3.0 · " + total["label"].lower(), f"{total['index_100']:.0f}")
        st.caption("100 — средний индекс исторической базы. Значение может быть выше 100.")
        st.write(explanation["summary"])
        st.caption("Это прогноз модели. Он сам по себе не определяет решение о запуске рекламы.")

    st.subheader("Что означают три оценки")
    st.write("У компонентов своя шкала: **позиция в базе сравнения от 0 до 100**. Чем выше число, тем выше оценка среди исторических примеров.")
    st.caption("Например, 7 — низкая позиция относительно базы. Это не означает, что посыл поняли 7% людей.")
    scored = [row for row in explanation["components"] if row["percentile"] is not None]
    if scored:
        st.plotly_chart(component_figure(scored), use_container_width=True,
                        config={"displayModeBar": False})
    st.caption("Низкий уровень: меньше 33 · Средний: от 33 до 67 · Высокий: от 67. Границы относятся к неокруглённым значениям.")
    with st.expander("Пояснения и вопросы по каждому компоненту"):
        for component in explanation["components"]:
            score_label = (f"{component['percentile']:.1f} из 100 · {component['level'].lower()}"
                           if component["percentile"] is not None else "Нет оценки")
            st.markdown(f"**{component['label']} · {score_label}**")
            st.write(component["outcome"])
            if component.get("source_observation"):
                st.write(component["source_observation"])
                st.caption("Это условная категория разметки ИИ. Время ответа людей не измерялось; отдельный признак не объясняет весь балл.")
            st.caption("Что проверить: " + component["inspect_question"])
    with st.expander("Как устроены шкалы и база сравнения"):
        st.write("AIPM 3.0 объединяет три компонента. Среднее значение итогового индекса в зафиксированной базе принято за 100. Компоненты и итоговый индекс нельзя складывать или сравнивать как одну шкалу.")
        st.write(f"Границы итоговых классов: ниже {AIPM3_Q33 * 100:.1f} — ниже среднего; от {AIPM3_Q33 * 100:.1f} до {AIPM3_Q67 * 100:.1f} — средний уровень; от {AIPM3_Q67 * 100:.1f} — выше среднего. Для компонентов границы — 33 и 67.")
        st.write("Каждый компонент сравнивается с заранее зафиксированными историческими оценками своей модели. База не меняется после загрузки нового ролика. Она не представляет все возможные рекламные кампании.")
        st.write("Заметность определяется одним из трёх классов модели, поэтому на её шкале возможны только три позиции: около 19, 55 и 86. Соседние числа и разные компоненты не следует трактовать как измерения одинаковой точности.")

    show_main_message(explanation["main_message"])
    if explanation["discrepancies"]:
        st.subheader("Как связать оценки с наблюдениями")
        for note in explanation["discrepancies"]:
            st.markdown("**" + note["title"] + "**")
            st.write(note["detail"])

    with st.expander("Наблюдения о ролике: звук, бренд, сюжет"):
        show_feature_profile(result, profile=profile)
    with st.expander("Насколько можно полагаться на результат"):
        for note in explanation["uncertainty_notes"]:
            st.write(note)
    return explanation, profile


def show_session_comparison(history: list[dict], *, scoring_version: str, protocol_version: str) -> None:
    compatible = [row for row in history
                  if row["scoring_version"] == scoring_version and row["protocol_version"] == protocol_version]
    if len(compatible) < 2:
        return
    st.subheader("Сравнение роликов в этой сессии")
    rows = [{
        "Ролик": row["name"], "AIPM 3.0 · среднее 100": round(row["index_100"]),
        "Заметность · 0–100": round(row["components"]["aipm1"]),
        "Запоминаемость · 0–100": round(row["components"]["aipm2"]),
        "Считываемость · 0–100": round(row["components"]["message_delivery"]),
        "Самая низкая позиция": row["lowest"],
    } for row in compatible]
    st.dataframe(rows, use_container_width=True, hide_index=True)
    st.caption("Близкие итоги могут сочетаться с разными слабыми компонентами. Разница в несколько пунктов сама по себе не доказывает превосходство одного ролика.")
    st.caption("Здесь хранятся последние 10 разных роликов этой сессии с одинаковыми версиями расчёта и анализа. Повторный анализ заменяет прежнюю строку. После перезапуска приложения история может пропасть — сохраните результаты.")
