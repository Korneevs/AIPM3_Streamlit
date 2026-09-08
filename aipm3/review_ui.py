"""Optional business diagnostics. These functions never call model scoring."""
from __future__ import annotations

import hashlib

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from . import creative_review as review


def show_manager_readout(result: dict) -> None:
    readout = review.manager_readout(result)
    st.subheader("Что поддерживает и что ограничивает оценку")
    st.info(readout["summary"])
    columns = st.columns(3)
    for column, (pillar, item) in zip(columns, readout["pillars"].items()):
        with column, st.container(border=True):
            st.markdown(f"**{pillar} · {item['level']} уровень**")
            for key, label in [("strength", "Поддерживает оценку"), ("limit", "Ограничивает оценку")]:
                group = item[key]
                if group:
                    st.markdown(f"**{label}: {group['Группа'].lower()}.**")
                    st.write(review.GROUP_MEANINGS.get(group["Группа"], ""))
                else:
                    st.caption(f"{label}: выраженный сигнал не выделен.")
    st.caption(
        "Это объяснение прогноза модели, а не доказательство причинного влияния на зрителя. "
        "Из него не следует, что изменение одного элемента обязательно повысит результат. "
        "Запоминаемость — модельная оценка; её прокси из линк-тестов требует подтверждения."
    )


def alignment_table(alignment: dict) -> pd.DataFrame:
    rows = []
    for dimension, label in review.EVALUATIONS.items():
        item = alignment["summary"][dimension]
        row = {"Что проверяем": label}
        for status, title in review.STATUS_LABELS.items():
            row[title] = f"{item['counts'][status]} из {item['n']}"
        rows.append(row)
    return pd.DataFrame(rows)


def _show_alignment(alignment: dict, key: str) -> None:
    st.markdown("#### Что из брифа выразили синтетические респонденты")
    st.dataframe(alignment_table(alignment), hide_index=True, use_container_width=True)
    if alignment["brief"]["uvp"] and alignment["brief"]["rtb"]:
        n = len(alignment["answers"])
        st.write(
            f"И выгода, и основание обещания выражены полностью в "
            f"{alignment['both_uvp_rtb_count']} из {n} ответов. "
            f"Логическая связь между ними выражена полностью в "
            f"{alignment['linked_uvp_rtb_count']} из {n} ответов."
        )
    st.caption(
        "Это число ответов синтетиков, не прогноз процента людей на линк-тесте. "
        "«Не выражено» означает отсутствие смысла в коротком ответе, а не доказанное непонимание. "
        "Упоминание RTB не доказывает доверие к нему или истинность обещания."
    )
    with st.expander("Проверить по исходным ответам"):
        rows = alignment["answers"]
        selected = st.selectbox("Ответ респондента", range(len(rows)),
                                format_func=lambda i: rows[i]["respondent_id"], key=key + "_answer")
        row = rows[selected]
        st.text(row["answer"] or "Ответ пустой")
        for dimension, label in review.EVALUATIONS.items():
            entry = row[dimension]
            st.markdown(f"**{label}: {review.STATUS_LABELS[entry['status']]}**")
            for quote in entry["quotes"]:
                st.text(quote)
            if entry["status"] == "uncertain":
                st.caption("Две разметки: " + " / ".join(review.STATUS_LABELS[s] for s in entry["coder_statuses"]))


def show_brief_review(result: dict, api_key: str) -> None:
    st.subheader("Считывается ли UVP и чем он подкреплён")
    st.write(
        "Сопоставьте бриф с тем, что уже сформулировали синтетические респонденты. "
        "UVP здесь — задуманный посыл кампании. При просмотре ролика респонденты не получали ни UVP, ни RTB."
    )
    answers = result.get("blind_answers", [])
    if not answers:
        st.info("В этом старом результате нет исходных ответов. Для проверки брифа запустите анализ ролика заново.")
        return
    diagnostics = result.setdefault("diagnostics", {})
    key = "brief_" + result.get("source_sha", result["video_sha"])
    saved = diagnostics.get("current_brief", {})
    brief = {
        "uvp": st.text_area("UVP — какой посыл хотели донести?", value=saved.get("uvp", ""),
                            max_chars=2000, key=key + "_uvp",
                            help="Главная мысль о ценности предложения, которую зритель должен вынести из ролика."),
        "rtb": st.text_area("RTB — чем подкрепляем обещание?", value=saved.get("rtb", ""),
                            max_chars=2000, key=key + "_rtb",
                            help="Факт, механизм, демонстрация или доказательство, объясняющие обещанную выгоду."),
    }
    brief = review.normalize_brief(brief)
    diagnostics["current_brief"] = brief
    alignment = diagnostics.get("alignment")
    current = review.alignment_is_current(alignment, brief, answers)
    st.caption("Можно заполнить часть брифа. Для проверки связи нужны и UVP, и RTB.")
    if st.button("Сопоставить с ответами", disabled=not any(brief.values()) or current, key=key + "_run"):
        try:
            with st.spinner("Сопоставляем записанные ответы с брифом. Оценки моделей не пересчитываются…"):
                alignment = review.compare_brief(brief, answers, api_key)
            diagnostics["alignment"] = alignment
            st.rerun()
        except Exception as exc:
            st.error(
                "Не удалось завершить сопоставление. Оценки ролика сохранены. "
                f"Попробуйте ещё раз. Тип ошибки: {type(exc).__name__}."
            )
    if current:
        _show_alignment(alignment, key)
    elif alignment:
        st.info("Бриф изменён. Предыдущее сопоставление к нему не относится — нажмите «Сопоставить с ответами».")
    with st.expander("Как устроена проверка брифа"):
        st.write(
            f"Используются {len(answers)} уже записанных ответов основной синтетической панели. "
            "Gemini отдельно кодирует их дважды по общей инструкции. Разногласия показываются явно; "
            "совпадение двух разметок не гарантирует правильность. Цитаты проверяются на наличие в исходных ответах. "
            "Бриф не передаётся в извлечение признаков или расчёт AIPM. При его изменении повторяется только кодирование. "
            "Основная идея в верхнем блоке определяется отдельной recovery-панелью на полном ролике."
        )


def scene_figure(scenes: list[dict]) -> go.Figure:
    figure = go.Figure(go.Bar(
        x=[s["end_seconds"] - s["start_seconds"] for s in scenes],
        base=[s["start_seconds"] for s in scenes],
        y=[f"Фрагмент {i + 1}" for i in range(len(scenes))],
        orientation="h", marker_color="#1EA7FD",
        hovertemplate="%{y}<br>Начало: %{base:.1f} сек.<br>Длительность: %{x:.1f} сек.<extra></extra>",
    ))
    figure.update_layout(height=max(210, len(scenes) * 38),
                         margin=dict(l=5, r=5, t=5, b=5),
                         xaxis_title="Секунды ролика", yaxis=dict(autorange="reversed"),
                         paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return figure


def show_scene_review(result: dict, api_key: str, video_bytes: bytes | None, suffix: str) -> None:
    with st.expander("Привязать разбор к сценам ролика"):
        st.caption(
            "Дополнительный просмотр без брифа: наблюдаемые элементы, цитаты и таймкоды. "
            "Он не изменяет признаки и оценки. Наблюдения Gemini нужно сверять с роликом."
        )
        diagnostics = result.setdefault("diagnostics", {})
        source_sha = result.get("source_sha")
        valid_video = bool(video_bytes and source_sha and hashlib.sha256(video_bytes).hexdigest() == source_sha)
        evidence = diagnostics.get("scene_evidence")
        if evidence and (evidence.get("source_sha") != source_sha or evidence.get("version") != review.VERSION):
            evidence = None
        if not valid_video:
            st.info("Для таймкодов загрузите тот же файл, для которого получена оценка.")
        if st.button("Разобрать сцены", disabled=not valid_video or evidence is not None, key="review_scenes"):
            try:
                with st.spinner("Просматриваем ролик и отмечаем наблюдаемые элементы…"):
                    evidence = review.review_uploaded_video(video_bytes, suffix, source_sha, api_key)
                diagnostics["scene_evidence"] = evidence
                st.rerun()
            except Exception as exc:
                st.error(f"Не удалось разобрать сцены. Оценки сохранены. Тип ошибки: {type(exc).__name__}.")
        if not evidence:
            return
        scenes = evidence["scenes"]
        if not scenes:
            st.info("В этом просмотре не получены подходящие наблюдения с таймкодами.")
            return
        st.plotly_chart(scene_figure(scenes), use_container_width=True)
        selected = st.selectbox("Фрагмент ролика", range(len(scenes)), key=f"review_scene_index_{source_sha}",
                                format_func=lambda i: f"{i + 1}: {scenes[i]['start_seconds']:g}–{scenes[i]['end_seconds']:g} сек.")
        scene = scenes[selected]
        st.text(scene["observation"])
        if "reported_end_seconds" in scene:
            st.caption("Округлённый таймкод конца ограничен фактической длительностью файла.")
        if scene["quote"]:
            st.text("Цитата: " + scene["quote"])
        st.caption("Группы признаков: " + "; ".join(review.GROUPS[g] for g in scene["groups"]))
        if valid_video:
            st.video(video_bytes, start_time=int(scene["start_seconds"]))
