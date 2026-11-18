"""Separate, session-scoped interface for creative scenario nominations."""
from __future__ import annotations

import csv
import hashlib
from html import escape
from io import StringIO
import json
import logging
import os

import streamlit as st

from aipm3.creative_inputs import CreativeInputError, MAX_FILE_BYTES, collect_scenarios, safe_csv_cell
from aipm3.creative_prompts import NOT_ASSESSED
from aipm3.creative_selection import DEFAULT_MODELS, PROTOCOL_VERSION, run_selection
from aipm3.latest_contracts import BASE_URL
from aipm3.runtime_resources import AnalysisBusy, AnalysisTimeout, analysis_deadline, analysis_slot


LOGGER = logging.getLogger(__name__)
STATUS_LABELS = {
    "agreed_initial": "Независимое согласие",
    "agreed_after_discussion": "Согласие после обсуждения",
    "rejected": "Не номинирован",
    "disputed": "Модели не согласились",
    "insufficient": "Недостаточно информации",
}
VERDICTS = {"nominate": "Номинировать", "reject": "Не номинировать", "insufficient": "Недостаточно информации"}
_CONFIRMED = {"agreed_initial", "agreed_after_discussion"}
_JUDGE_LABELS = ("GPT 6.1 Sol", "Claude Opus 5.5")
_PASTE_EXAMPLE = "=== Название первого сценария ===\nПолный текст первого сценария.\n\n=== Название второго сценария ===\nПолный текст второго сценария."


def api_key() -> str:
    try:
        return st.secrets.get("VSELLM_API_KEY") or os.environ.get("VSELLM_API_KEY", "")
    except Exception:
        return os.environ.get("VSELLM_API_KEY", "")


def identity(scenarios, quota, models):
    payload = {"user": st.session_state.get("mm_user"), "scenarios": scenarios,
               "quota": quota, "models": list(models), "protocol": PROTOCOL_VERSION}
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def analyze(scenarios, quota, key, models, queued, progress):
    with analysis_slot(wait=True, on_queue=queued), analysis_deadline():
        return run_selection(scenarios=scenarios, quota=quota, api_key=key,
                             api_base=BASE_URL, models=models, on_progress=progress)


def _paragraph(label, value):
    if value and value != NOT_ASSESSED:
        st.html(f"<p><strong>{escape(label)}</strong> {escape(str(value)).replace(chr(10), '<br>')}</p>")


def _opinion(opinion, *, label):
    st.markdown(f"**{label} · {VERDICTS.get(opinion.get('verdict'), 'Нет решения')}**")
    for key, title in (("reason", "Основание:"), ("quote", "Эпизод:"),
                       ("mechanism", "Приём:"),
                       ("decision_change_reason", "После обсуждения:")):
        _paragraph(title, opinion.get(key))


def show_card(row, *, expanded=False):
    with st.container(border=True):
        st.html(f"<h4>{escape(row['title'])}</h4>")
        st.caption(STATUS_LABELS.get(row["status"], row["status"]))
        for key, title in (("idea", "Идея:"), ("quote", "Сильный эпизод:"),
                           ("mechanism", "За счёт чего работает:"),
                           ("distinctiveness", "Что добавляет:"), ("overflow_reason", "Почему сверх ориентира:")):
            _paragraph(title, row.get(key))
        if row["status"] == "disputed":
            for label, opinion in zip(_JUDGE_LABELS, row.get("final", row.get("initial", []))):
                _opinion(opinion, label=label)
        else:
            _paragraph("Рекламная ценность:", row.get("reason"))
        with st.expander("Полный сценарий", expanded=expanded):
            st.text(row["text"])
        with st.expander("Решения двух моделей"):
            for index, opinion in enumerate(row.get("initial", []), 1):
                _opinion(opinion, label=f"Оценщик {index} · первый проход")
            if row.get("discussion_happened"):
                for index, opinion in enumerate(row.get("final", []), 1):
                    _opinion(opinion, label=f"Оценщик {index} · после обсуждения")


def report_csv(result) -> bytes:
    output = StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow(["ID", "Сценарий", "Решение", "Основная идея", "Сильный эпизод", "Творческий приём",
                     "Рекламная ценность", "Что добавляет",
                     "Почему сверх ориентира", "Первый проход", "После обсуждения", "Полный текст"])
    for row in result["items"]:
        opinions = lambda key: "; ".join(f"{i + 1}: {VERDICTS.get(o.get('verdict'), '')}" for i, o in enumerate(row.get(key, [])))
        values = [row.get("id"), row.get("title"), STATUS_LABELS.get(row["status"], row["status"]),
                  *[row.get(key, "") for key in ("idea", "quote", "mechanism", "reason", "distinctiveness", "overflow_reason")],
                  opinions("initial"), opinions("final") if row.get("discussion_happened") else "", row.get("text")]
        writer.writerow([safe_csv_cell(value) for value in values])
    return output.getvalue().encode("utf-8-sig")


def show_result(result):
    rows = result["items"]
    by_id = {row["id"]: row for row in rows}
    primary_ids, additional_ids = result.get("primary_ids", []), result.get("additional_ids", [])
    confirmed = [row for row in rows if row["status"] in _CONFIRMED]
    disputed = [row for row in rows if row["status"] == "disputed"]
    insufficient = [row for row in rows if row["status"] == "insufficient"]
    metrics = st.columns(3)
    metrics[0].metric("Сценариев рассмотрено", len(rows))
    metrics[1].metric("Прошедших сценариев", len(confirmed))
    metrics[2].metric("Спорных сценариев", len(disputed))
    st.subheader("Прошедшие отбор")
    if not confirmed:
        st.info("Согласованных креативных кандидатов не найдено. Места не заполняются автоматически.")
    for key in primary_ids:
        show_card(by_id[key])
    if additional_ids:
        st.subheader("Дополнительные идеи")
        st.write(f"Найдено {len(primary_ids) + len(additional_ids)} самостоятельных идей при ориентире {result['quota']}. "
                 "Ниже — обоснование каждого дополнительного кандидата.")
        for key in additional_ids:
            show_card(by_id[key])
    variants = [row for row in confirmed if row["id"] not in set(primary_ids + additional_ids)]
    if variants:
        with st.expander(f"Близкие варианты выбранных идей · {len(variants)}"):
            for row in variants:
                group = next((g for g in result.get("groups", []) if row["id"] in g.get("member_ids", [])), {})
                representative = by_id.get(group.get("representative_id"), {})
                if representative:
                    _paragraph("Близок к:", representative.get("title"))
                    _paragraph("Почему выбран представитель:", group.get("representative_reason"))
                show_card(row)
    if disputed:
        st.subheader("Спорные кандидаты")
        st.write("После одного обмена аргументами модели не согласились. Решение — за вами.")
        for row in disputed:
            show_card(row)
    if insufficient:
        st.subheader("Нужна информация")
        for row in insufficient:
            with st.expander(row["title"]):
                for index, opinion in enumerate(row.get("final", row.get("initial", [])), 1):
                    _opinion(opinion, label=f"Оценщик {index}")
                st.text(row["text"])
    rejected = [row for row in rows if row["status"] == "rejected"]
    if rejected:
        with st.expander(f"Не номинированы · {len(rejected)}"):
            for row in rejected:
                st.html(f"<h4>{escape(row['title'])}</h4>")
                for index, opinion in enumerate(row.get("final", row.get("initial", [])), 1):
                    _opinion(opinion, label=f"Оценщик {index}")
    st.caption("Рекламная ценность — гипотеза по сценарию, а не измеренный эффект.")
    downloads = st.columns(2)
    downloads[0].download_button("Скачать таблицу CSV", report_csv(result), "creative_selection.csv", "text/csv", key="creative_download_csv")
    downloads[1].download_button("Скачать полный разбор JSON", json.dumps(result, ensure_ascii=False, indent=2),
                                 "creative_selection.json", "application/json", key="creative_download_json")


def main():
    st.title("Креативный отбор")
    st.write("Найдите выделяющиеся идеи с понятной рекламной ценностью: вниманием к предложению, "
             "запоминанием продукта или интересом воспользоваться. Две модели независимо рассмотрят каждый сценарий.")
    st.caption("GPT 6.1 Sol + Claude Opus 5.5")
    uploads = st.file_uploader("Сценарии", type=["txt", "md", "docx", "csv"], accept_multiple_files=True,
                               help="Один файл — один сценарий; в CSV нужны title и text, id необязателен. До 60 сценариев, каждый файл до 4 МБ.", key="creative_uploads")
    pasted = st.text_area("Или вставьте сценарии", height=160, placeholder=_PASTE_EXAMPLE, key="creative_pasted",
                          help="Разделяйте сценарии строкой === Название ===. Без разделителей текст считается одним сценарием.")
    quota = int(st.number_input("Ориентир числа идей для просмотра", min_value=1, max_value=20, value=2,
                               help="Отбор может вернуть ноль идей или больше ориентира. Каждая дополнительная идея требует обоснования.", key="creative_quota"))
    with st.expander("Как проходит отбор"):
        st.write("Сначала две модели независимо оценивают силу творческого решения и его рекламную ценность. "
                 "Бриф и скоры считываемости не учитываются. Наличие приёма без сильного решения и понятной пользы не даёт проход. "
                 "При разногласии они один раз проверяют аргументы друг друга. Сохранившиеся разногласия показываются отдельно. "
                 "Близкие идеи объединяются; превышение ориентира объясняется по каждому дополнительному кандидату.")
    scenarios, input_error = [], ""
    try:
        if any(getattr(file, "size", 0) > MAX_FILE_BYTES for file in (uploads or [])):
            raise CreativeInputError("Один из файлов больше 4 МБ. Загрузите только текст сценария.")
        scenarios = collect_scenarios([(file.name, file.getvalue()) for file in (uploads or [])], pasted)
    except CreativeInputError as err:
        input_error = str(err)
        st.warning(input_error)
    if scenarios:
        st.caption(f"Готово к сравнению: {len(scenarios)} сценариев.")
        with st.expander("Проверить загруженные тексты"):
            for row in scenarios:
                st.html(f"<h4>{escape(row['title'])}</h4>")
                st.text(row["text"])
    ready = bool(scenarios and not input_error)
    run = st.button("Найти креативные идеи", type="primary", disabled=not ready, key="creative_run")
    if not ready:
        return
    models = tuple(DEFAULT_MODELS)
    fingerprint = identity(scenarios, quota, models)
    retry_failed = False
    if run:
        key = api_key()
        if not key:
            retry_failed = True
            st.error("Анализ временно недоступен. Обратитесь к администратору приложения.")
        else:
            notice = st.empty()
            with st.status("Готовим отбор", expanded=True) as status:
                def queued(position):
                    if position:
                        notice.warning(f"Сервер занят. Ваш номер в очереди — {position}. Анализ начнётся автоматически.")
                        status.update(label="Ожидаем своей очереди")
                    else:
                        notice.empty()
                try:
                    result = analyze(scenarios, quota, key, models, queued, lambda message: status.update(label=message))
                except (AnalysisBusy, AnalysisTimeout):
                    retry_failed = True
                    status.update(label="Время ожидания истекло", state="error")
                    st.warning("Повторите запуск позже.")
                except Exception as err:
                    retry_failed = True
                    LOGGER.warning("Creative selection failed (%s)", type(err).__name__)
                    status.update(label="Не удалось завершить отбор", state="error")
                    st.error("Не удалось получить полный разбор от обеих моделей. Повторите запуск. "
                             "Незавершённый отбор не считается отказом от сценариев.")
                else:
                    st.session_state["creative_result"] = (fingerprint, result)
                    status.update(label="Отбор завершён", state="complete", expanded=False)
                finally:
                    notice.empty()
    saved = st.session_state.get("creative_result")
    if saved and saved[0] == fingerprint:
        if retry_failed:
            st.caption("Ниже — предыдущий завершённый разбор этих же сценариев.")
        show_result(saved[1])


if __name__ in {"__page__", "__main__"}:
    main()
