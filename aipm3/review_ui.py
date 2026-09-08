"""Optional business diagnostics. These functions never call model scoring."""
from __future__ import annotations

import hashlib

import plotly.graph_objects as go
import streamlit as st

from . import creative_review as review
from .creative_diagnosis import build_diagnosis


def show_manager_readout(result: dict) -> None:
    report = build_diagnosis(result)
    st.subheader("Разбор ролика: что работает и где теряется смысл")
    st.write(report["headline"])
    for column, section, title in zip(st.columns(2), ["strengths", "risks"],
                                      ["Что получилось", "Что ослабляет коммуникацию"]):
        with column:
            st.markdown("#### " + title)
            if not report[section]:
                st.caption("Устойчивых оснований для однозначного вывода не выделено.")
            for finding in report[section]:
                with st.container(border=True):
                    st.markdown("**" + finding["title"] + "**")
                    st.write(finding["meaning"])
                    scene = finding["scene"]
                    if scene:
                        st.caption(f"В ролике · примерно {scene['start_seconds']:g}–{scene['end_seconds']:g} сек.")
                        st.text("«" + scene["quote"] + "»")
                    if finding.get('answer_example'):
                        st.caption("Пример ответа после полного просмотра")
                        st.text("«" + finding['answer_example'] + "»")
                    if finding.get('transcript_quote'):
                        st.caption("Ключевая реплика из сохранённых транскриптов")
                        st.text("«" + finding['transcript_quote'] + "»")
                    with st.expander("На чём основан вывод"):
                        st.write(finding["evidence"])
                        if finding["signal"]:
                            st.caption("Признак: " + finding["signal"]["feature"])
    if report["uncertain"]:
        st.markdown("**Где нет уверенного вывода**")
        for text in report["uncertain"]:
            st.write(text)
    if not report["has_panel"]:
        st.caption("В сохранённом результате нет повторных оценок диагностических признаков. "
                   "Показаны только выводы, для которых есть основания; расширенная разметка сохранится при следующем анализе.")
    with st.expander("Как читать этот разбор"):
        st.write("Это диагностика коммуникации, не список рекомендаций и не вклад каждого элемента в итоговый балл. "
                 "Используются уже извлечённые признаки, включая диагностические признаки вне финального регрессора. "
                 "Для оценок панели требуется согласие не менее 80% ответов и 80% запросов. "
                 "30 ответов получены в 10 запросах, это не 30 независимых людей. "
                 "Пороги интерпретации — фиксированные правила, а не новые валидированные нормативы.")
        st.write("На одинаковой сохранённой разметке выводы одинаковы. Это не гарантирует совпадения при новом просмотре Gemini. "
                 "Цитаты из сцен показываются только при подтверждении минимум двумя транскриптами; таймкоды приблизительные. "
                 "Сильная сторона не гарантирует высокий общий балл. Замечание о риске не доказывает причинного влияния на людей.")


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
