"""Optional business diagnostics. These functions never call model scoring."""
from __future__ import annotations

import hashlib

import plotly.graph_objects as go
import streamlit as st

from . import creative_review as review
from . import brief_details
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


def _dimension_story(alignment, dimension):
    item = alignment["summary"][dimension]
    counts, n = item["counts"], item["n"]
    if counts["not_set"] == n:
        return "Не проверяли: этот элемент не задан."
    if counts["uncertain"] > n / 3:
        return "Разметки заметно расходятся. Однозначный вывод пока ненадёжен."
    if counts["contradicted"] > n / 2:
        return "В большинстве ответов появляется противоположный смысл."
    if counts["matched"] >= .8 * n:
        return "Заданный смысл сохраняется в большинстве пересказов."
    if counts["partial"] > n / 2:
        return "В ответах сохраняется лишь часть задуманного смысла."
    if counts["absent"] > n / 2:
        return "В большинстве пересказов этот смысл явно не выражен."
    return "Смысл сохраняется не во всех ответах: есть и точные, и неполные пересказы."


def _show_alignment(alignment: dict, key: str, details=None) -> None:
    st.markdown("#### Что осталось от задуманного посыла")
    if details:
        losses = []
        for part in details['summary']:
            for phrase in sorted({r.get('missing_brief_quote') for r in part['answers'] if r.get('missing_brief_quote')}):
                n = sum(r.get('missing_brief_quote') == phrase for r in part['answers'])
                if n >= .5 * part['n']:
                    losses.append((n, part['n'], phrase))
        if losses:
            count, total, phrase = sorted(losses, key=lambda x: (-x[0], x[2]))[0]
            st.warning(f"Важная потеря в пересказе: «{phrase}». В {count} из {total} ответов эта деталь "
                       "не сохраняется, хотя остальная часть предложения выражена.")
    for column, (dimension, label) in zip(st.columns(2), [("uvp", "Задуманный посыл"), ("rtb", "Основание обещания")]):
        if not alignment["brief"][dimension]:
            continue
        with column, st.container(border=True):
            st.caption(label.upper() + " · " + alignment["brief"][dimension])
            st.markdown("**" + _dimension_story(alignment, dimension) + "**")
            counts = alignment["summary"][dimension]["counts"]
            primary = max(["matched", "partial", "absent", "contradicted", "uncertain"], key=lambda s: counts[s])
            examples = [r for r in alignment["answers"] if r[dimension]["status"] == primary and r["answer"].strip()]
            if examples:
                example = sorted(examples, key=lambda r: (len(r["answer"]), r["respondent_id"]))[len(examples) // 2]
                st.caption("Пример " + ("неполного пересказа" if primary == "partial" else "ответа"))
                st.text("«" + example["answer"] + "»")
            st.caption(f"Из {alignment['summary'][dimension]['n']} ответов: полностью — {counts['matched']}; "
                       f"частично — {counts['partial']}; не выражено — {counts['absent']}; "
                       f"противоположный смысл — {counts['contradicted']}; спорная разметка — {counts['uncertain']}.")
    if details:
        st.markdown("**Какие детали сохраняются, а какие теряются**")
        for part in details["summary"]:
            c, n = part["counts"], part["n"]
            if c["matched"] >= .8 * n:
                label, comment = "Сохраняется", f"выражено полностью в {c['matched']} из {n} ответов"
            elif c["absent"] >= .8 * n:
                label, comment = "Теряется в пересказе", f"не выражено в {c['absent']} из {n} ответов"
            elif c["contradicted"] > n / 2:
                label, comment = "Меняет смысл", f"противоположный смысл в {c['contradicted']} из {n} ответов"
            elif c["uncertain"] > n / 3:
                label, comment = "Пока спорно", f"разметки расходятся в {c['uncertain']} из {n} ответов"
            else:
                label, comment = "Сохраняется не всегда", f"полностью — {c['matched']}, частично — {c['partial']} из {n} ответов"
            st.write(f"{label}: «{part['brief_quote']}» — {comment}.")
            missing = sorted({r['missing_brief_quote'] for r in part['answers'] if r.get('missing_brief_quote')})
            for phrase in missing:
                cases = [r for r in part['answers'] if r.get('missing_brief_quote') == phrase]
                if len(cases) >= .5 * n:
                    st.markdown(f"**Что именно потерялось: «{phrase}»** — в {len(cases)} из {n} ответов.")
                    st.text("Пример: «" + cases[0]['answer'] + "»")
                    st.caption("Оба кодировщика согласились: остальная часть детали есть, но этот смысл не выражен.")
    if alignment["brief"]["uvp"] and alignment["brief"]["rtb"]:
        n = len(alignment["answers"])
        st.markdown("**Работают ли UVP и RTB вместе?**")
        if alignment['linked_uvp_rtb_count'] >= .8 * n:
            st.write("В большинстве ответов основание связано с выгодой: сохраняется не только обещание, но и объяснение, на чём оно держится.")
        else:
            st.write("Упоминания предложения и его основания ещё не складываются в устойчивый ответ «почему именно этот сервис даёт эту выгоду». "
                     "Короткий ответ может опускать эту связь, даже если она понятна при просмотре.")
        st.caption(f"Полная смысловая связь выражена в {alignment['linked_uvp_rtb_count']} из {n} ответов. "
                   "Просто соседние упоминания не считаются объяснением.")
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
    with st.expander("Бриф кампании", expanded=not bool(diagnostics.get("alignment"))):
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
    details = diagnostics.get("brief_details")
    details_current = brief_details.is_current(details, brief, answers)
    st.caption("Можно заполнить часть брифа. Для проверки связи нужны и UVP, и RTB.")
    if st.button("Сопоставить с ответами", disabled=not any(brief.values()) or current, key=key + "_run"):
        try:
            with st.spinner("Сопоставляем записанные ответы с брифом. Оценки моделей не пересчитываются…"):
                alignment = review.compare_brief(brief, answers, api_key)
            diagnostics["alignment"] = alignment
            try:
                with st.spinner("Проверяем, какие отдельные детали брифа сохраняются в пересказах…"):
                    diagnostics["brief_details"] = brief_details.compare(brief, answers, api_key)
                diagnostics.pop("brief_details_error", None)
            except Exception as exc:
                diagnostics["brief_details_error"] = type(exc).__name__
            st.rerun()
        except Exception as exc:
            st.error(
                "Не удалось завершить сопоставление. Оценки ролика сохранены. "
                f"Попробуйте ещё раз. Тип ошибки: {type(exc).__name__}."
            )
    if current:
        _show_alignment(alignment, key, details if details_current else None)
        if not details_current:
            st.caption("Проверка брифа готова. Детальный разбор смысловых потерь ещё не рассчитан.")
            if st.button("Разобрать отдельные детали посыла", key=key + "_details"):
                try:
                    with st.spinner("Проверяем детали по сохранённым ответам — без пересчёта ролика…"):
                        diagnostics["brief_details"] = brief_details.compare(brief, answers, api_key)
                    diagnostics.pop("brief_details_error", None)
                    st.rerun()
                except Exception as exc:
                    st.error("Не удалось разобрать детали. Основные результаты сохранены. "
                             f"Тип ошибки: {type(exc).__name__}.")
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
        st.write("Для подробного разбора бриф делится на смысловые детали. Каждая — цитата из введённого брифа; "
                 "их наличие в ответах независимо кодируется дважды с учётом перефразирования. "
                 "Список деталей фиксируется вместе с результатом; спорная разметка не превращается в уверенный вывод. "
                 "Это диагностическое кодирование, его точность на человеческих ответах отдельно не валидирована.")


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
