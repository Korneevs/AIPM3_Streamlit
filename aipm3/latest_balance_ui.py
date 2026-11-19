"""A compact, additive explanation of the current per-component score."""
from __future__ import annotations

from html import escape
from textwrap import wrap

import plotly.graph_objects as go
import streamlit as st

from .latest_interpretation import LABELS

TITLES = {"n": "Заметность", "m": "Считываемость", "r": "Запоминаемость"}


def driver_label(driver):
    if (driver["feature"] == "panel__offer_novelty_explanation_need"
            and driver["value"] < .5 and driver.get("evidence", {}).get("verified")):
        return "Предложение почти не требует пояснений"
    return driver["label"]


def displayable_driver(interpretation, task, driver):
    """Respect the applied editorial review without changing numerical effects."""
    if not driver.get("usable") or not driver.get("evidence", {}).get("verified"):
        return False
    if interpretation.get("material_kind") == "neuromatics":
        from .latest_runtime import load_models
        neutralized = getattr(load_models("neuromatics").heads[task], "neutralized_features", ())
        if driver["feature"] in neutralized:
            return False
    semantic = interpretation.get("manager_semantic") or {}
    matched = False
    for row in semantic.get("profiles", {}).get(task, []):
        if driver["feature"] not in row.get("features", [row.get("feature")]):
            continue
        matched = True
        fact = row.get("evidence_facts", {}).get(driver["feature"])
        if row.get("status") != "supported" or not fact or not fact.get("same_input"):
            return False
        if semantic.get("reviewed") and (
                row.get("review_action") == "withhold"
                or driver["feature"] not in row.get("reviewed_support_ids", [])):
            return False
        if semantic.get("reviewed") and row.get("direction") in {"up", "down"}:
            sign = "up" if driver["contribution"] > 0 else "down" if driver["contribution"] < 0 else "balanced"
            if row["direction"] != sign:
                return False
    return matched or not semantic.get("reviewed")


def balance_items(interpretation, task, limit=4):
    detail = interpretation["details"][task]
    selected = sorted((d for d in detail["drivers"]
                       if displayable_driver(interpretation, task, d)),
                      key=lambda d: abs(d["contribution"]), reverse=True)[:limit]
    adjustment = float(detail.get("manual_adjustment", 0.0))
    rest = (detail["actual"] - detail["expected"] - adjustment
            - sum(d["contribution"] for d in selected))
    items = [dict(label="База", value=detail["expected"], measure="absolute")]
    items.extend(dict(label=driver_label(d), value=d["contribution"], measure="relative",
                      driver=d) for d in selected)
    items.append(dict(label="Остальные факторы", value=rest, measure="relative"))
    if adjustment:
        items.extend([
            dict(label="Оценка модели", value=detail["actual"] - adjustment, measure="total"),
            dict(label="Поправка за известного человека", value=adjustment, measure="relative",
                 expert=True),
        ])
    items.append(dict(label="Итог", value=detail["actual"], measure="total"))
    return items


def explanation_is_partial(interpretation, task, card):
    if card.get("explanation_status") == "partial":
        return True
    # The editor can withhold a material explanation while numerical SHAP
    # and the original card stay unchanged. A top-four cut alone is harmless.
    return any(driver.get("usable") and not displayable_driver(interpretation, task, driver)
               for driver in interpretation["details"][task]["drivers"])


def balance_figure(interpretation, task):
    items = balance_items(interpretation, task)
    wrapped = [wrap(item["label"], 24) for item in items]
    labels = ["<br>".join(escape(part) for part in parts) for parts in wrapped]
    text = [(f'{item["value"]:+.4f}' if item["measure"] == "relative"
             else f'{item["value"]:.4f}').replace(".", ",") for item in items]
    figure = go.Figure(go.Waterfall(
        orientation="h", y=labels,
        x=[0.0 if item["measure"] == "total" else item["value"] for item in items],
        measure=[item["measure"] for item in items], text=text,
        textposition="outside", cliponaxis=False,
        increasing=dict(marker=dict(color="#23875e", line=dict(width=0))),
        decreasing=dict(marker=dict(color="#b94d59", line=dict(width=0))),
        totals=dict(marker=dict(color="#7e8997", line=dict(width=0))),
        connector=dict(line=dict(color="#7e8997", width=1)),
        hovertemplate="%{y}<br>%{text}<extra></extra>",
    ))
    figure.update_layout(
        height=95 + max(43, max(map(len, wrapped)) * 16 + 12) * len(items),
        margin=dict(l=0, r=65, t=12, b=40),
        showlegend=False, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title="Балл модели", fixedrange=True, rangemode="tozero", zeroline=False),
        yaxis=dict(autorange="reversed", fixedrange=True, automargin=True),
        waterfallgap=.35,
    )
    return figure


def show_balance_profile(interpretation):
    st.subheader("Баланс оценки")
    cards = {card["task"]: card for card in interpretation["cards"]}
    for tab, task in zip(st.tabs(list(TITLES.values())), TITLES):
        with tab:
            card = cards[task]
            if not card.get("assessment_available", True):
                st.caption("Для оценки нужна полная озвучка.")
                continue
            st.markdown(f'**{card["score"]:.4f} · класс {card["class"]}**'.replace(".", ","))
            items = balance_items(interpretation, task)
            st.plotly_chart(balance_figure(interpretation, task), use_container_width=True,
                            config={"displayModeBar": False}, key="score_balance_" + task)
            st.caption("Вклад в расчёт, а не прогноз результата правки.")
            if interpretation["details"][task].get("manual_adjustment"):
                with st.expander("Участие известного человека"):
                    st.write("Участие отмечено вручную. К оценке запоминаемости применена "
                             "экспертная поправка +20%. Она показана отдельно от вкладов модели; "
                             "это не измеренный эффект участия человека в этом ролике.")
            with st.expander("На чём основаны вклады"):
                base = f'{interpretation["details"][task]["expected"]:.4f}'.replace(".", ",")
                reference = f'{card["baseline"]:.4f}'.replace(".", ",")
                st.caption(f"База графика: {base}. Ориентир 0% в верхней карточке: {reference}. "
                           "Это разные точки отсчёта. База и вклады показаны в тех же баллах, что и итог.")
                for item in items:
                    driver = item.get("driver")
                    if driver is None:
                        continue
                    st.markdown(f'**{driver_label(driver)} · {driver["contribution"]:+.4f}**')
                    evidence = driver.get("evidence", {})
                    st.write(evidence.get("observation", ""))
                    for episode in evidence.get("episodes", [])[:2]:
                        if episode.get("start") is not None and episode.get("end") is not None:
                            st.caption(f'{episode["start"]:g}–{episode["end"]:g} с: '
                                       + episode.get("description", ""))
                    if str(evidence.get("source", "")).startswith("physical"):
                        st.caption("Автоматическое измерение файла.")
                    else:
                        st.caption("Наблюдение из повторных AI-разборов; ручная проверка не выполнялась.")
                    if driver.get("interpretation_kind") == "association_only":
                        st.caption("Особенность расчёта этого ролика: знак не означает, "
                                   "что приём нужно добавить или убрать.")
                st.caption("Остальные факторы — сумма всех вкладов, не показанных отдельно: "
                           "за пределами четырёх крупнейших, небольших, нестабильных "
                           "или не подтверждённых наблюдениями, а также вкладов, "
                           "пояснения к которым не приняты проверкой.")
            if explanation_is_partial(interpretation, task, card):
                st.caption("Разбор частичный: не все выводы подтверждены проверкой.")
            with st.expander("Все численные вклады"):
                st.dataframe([
                    {"Признак": (driver_label(d) if displayable_driver(interpretation, task, d)
                                 else LABELS.get(d["feature"], d["label"])), "Вклад": d["contribution"],
                     "Наблюдение подтверждено": bool(d.get("evidence", {}).get("verified")),
                     "Допущен к объяснению": displayable_driver(interpretation, task, d)}
                    for d in interpretation["details"][task]["drivers"]
                ], hide_index=True, use_container_width=True)
