"""Main pretest page for the frozen latest model and evidence-based review."""
from __future__ import annotations

import hashlib
from html import escape
import json
import os
from pathlib import Path
import tempfile

import streamlit as st

from aipm3.latest_interpretation import VERSION, build_latest_interpretation, public_result
from aipm3.latest_pipeline import MissingMeasurement, run_latest_analysis
from aipm3.latest_runtime import (
    ANALYSIS_REPEATS, MATERIAL_LABELS, clean_json, material_kind_for_result, validate_cached_result,
)
from aipm3.runtime_resources import AnalysisBusy
from aipm3.manager_report import LEVELS, report_cards
from aipm3.display_calibration import audio_status
from aipm3.latest_manual_inputs import celebrity_presence, with_celebrity_review


VERTICALS = {"Товары": "Goods", "Авто": "Auto", "Работа": "Jobs",
             "Недвижимость": "Realty", "Услуги": "Services", "Путешествия": "Travel"}


@st.cache_data(show_spinner=False)
def _interpret_cached(result: dict, version: str) -> dict:
    return build_latest_interpretation(result, evidence=result.get("independent_evidence", []))


def interpret_result(result: dict) -> dict:
    return _interpret_cached(result, VERSION)


def _show_score(title: str, index: float, level: str, available: bool = True) -> None:
    label, color, background = LEVELS[level]
    with st.container(border=True):
        if not available:
            st.metric(title, "—")
            st.caption("Нужна полная озвучка")
            return
        st.metric(title, f"{index:.0f}")
        st.markdown(f'<div style="background:{background};color:{color};padding:8px 12px;'
                    f'border-radius:7px;font-weight:650">{escape(label)}</div>',
                    unsafe_allow_html=True)


def _show_driver(item: dict) -> None:
    st.markdown(f'**{item["label"]}**')
    st.write(item["takeaway"])
    if item.get("evidence"):
        st.caption("Основание: " + item["evidence"])
    if item.get("episodes"):
        st.caption("В ролике: " + "; ".join(
            f'{float(ep["start"]):g}-{float(ep["end"]):g} с' for ep in item["episodes"]))
    if item.get("check"):
        st.write(item["check"])


def show_latest_result(result: dict, interpretation: dict | None = None) -> None:
    material_kind = material_kind_for_result(result)
    interpretation = interpretation or interpret_result(result)
    overall = interpretation.get("overall")
    if interpretation.get("audio_note"):
        st.warning(interpretation["audio_note"])
    if overall:
        st.subheader("Общая оценка")
        _show_score("AIPM3.0", overall["index"], overall["level"], overall.get("assessment_available", True))
    if not overall or overall.get("assessment_available", True):
        st.caption("100 - ориентир для нейроматиков, сопоставленный с готовыми роликами." if material_kind == "neuromatics"
                   else "100 - средняя оценка роликов, с которыми сравниваем этот вариант.")
    for column, card in zip(st.columns(3), interpretation["cards"]):
        with column:
            _show_score(card["title"], card["index"], card["level"], card.get("assessment_available", True))

    cards = report_cards(interpretation)
    for card in cards:
        with st.container(border=True):
            st.subheader(card["title"])
            st.write(card["description"])
            st.write(card["summary"])
            if card["strengths"]:
                st.markdown("**Что поддерживает оценку**")
                for item in card["strengths"]:
                    _show_driver(item)
            if card["limitations"]:
                st.markdown("**Что ограничивает оценку**")
                for item in card["limitations"]:
                    _show_driver(item)
            if card["unassessed"]:
                st.markdown("**Что нельзя уверенно объяснить**")
                for item in card["unassessed"]:
                    st.markdown("**" + "; ".join(item["features"]) + "**")
                    st.write(item["text"])
    st.caption("Оценка помогает сравнивать варианты. Финальное решение принимает Марком. "
               "Предложения по правкам стоит проверить на следующей версии ролика.")
    exported = clean_json(public_result(result, interpretation))
    exported["manager_cards"] = clean_json(cards)
    name = Path(result.get("source_name", "ролик")).stem
    st.download_button("Скачать разбор", data=json.dumps(
        exported, ensure_ascii=False, indent=2, allow_nan=False),
        file_name=f"{name}_анализ.json", mime="application/json",
        key=f"latest_{material_kind}_download")


def _api_key() -> str:
    try:
        return str(st.secrets.get("VSELLM_API_KEY", "")) or os.environ.get("VSELLM_API_KEY", "")
    except Exception:
        return os.environ.get("VSELLM_API_KEY", "")


def _review_collector():
    # The corrected evidence protocol is independent of the scoring inputs.
    try:
        from aipm3.latest_evidence import collect_full_evidence
    except ModuleNotFoundError as exc:
        if exc.name != "aipm3.latest_evidence":
            raise
        return None
    return collect_full_evidence


def main(material_kind: str = "finished") -> None:
    if material_kind not in MATERIAL_LABELS:
        raise ValueError(f"Unknown material kind: {material_kind}")
    prefix = f"latest_{material_kind}_"
    result_key = prefix + "result"
    st.title(MATERIAL_LABELS[material_kind])
    st.caption("Что поддерживает оценку ролика, что её ограничивает и какие выводы пока нельзя сделать.")
    mode = st.radio("Источник результата", ["Загрузить ролик", "Открыть сохранённый результат"],
                    horizontal=True, key=prefix + "source")
    preset_env = "AIPM_LATEST_RESULT_JSON" if material_kind == "finished" else "AIPM_NEUROMATICS_RESULT_JSON"
    preset = os.environ.get(preset_env)
    if preset and st.session_state.get(prefix + "preset") != preset:
        try:
            st.session_state[result_key] = validate_cached_result(
                json.loads(Path(preset).read_text()), material_kind=material_kind)
            st.session_state[prefix + "preset"] = preset
        except (ValueError, KeyError, OSError):
            st.error("Не удалось открыть сохранённый разбор для этого типа роликов.")

    if mode == "Открыть сохранённый результат":
        saved = st.file_uploader("Файл с результатом анализа", type=["json"], key=prefix + "saved")
        if saved is not None:
            digest = hashlib.sha256(saved.getvalue()).hexdigest()
            if st.session_state.get(prefix + "saved_sha") != digest:
                try:
                    st.session_state[result_key] = validate_cached_result(
                        json.loads(saved.getvalue()), material_kind=material_kind)
                    st.session_state[prefix + "saved_sha"] = digest
                except (ValueError, KeyError, TypeError):
                    st.error("Не удалось открыть файл. Выберите разбор, сохранённый для этого типа роликов.")
                    st.stop()
        st.caption("Готовый разбор открывается без повторного анализа ролика.")
    else:
        vertical = st.selectbox("Вертикаль", list(VERTICALS), key=prefix + "vertical")
        uploaded = st.file_uploader("Ролик Avito (MP4 / MOV)", type=["mp4", "mov"], key=prefix + "video")
        if uploaded is not None:
            st.video(uploaded)
        cached, live = st.columns(2)
        cached_btn = cached.button("Использовать прошлый анализ", use_container_width=True,
                                   key=prefix + "cached")
        live_btn = live.button("Проанализировать ролик", type="primary", use_container_width=True,
                               key=prefix + "live")
        st.caption("Анализ и проверка наблюдений могут занять несколько минут.")
        if cached_btn or live_btn:
            if uploaded is None:
                st.warning("Сначала загрузите ролик.")
            elif live_btn and not _api_key():
                st.error("Новый анализ пока недоступен. Обратитесь к администратору приложения.")
            else:
                cache = Path(os.environ.get("AIPM_LATEST_CACHE_DIR",
                                            str(Path(tempfile.gettempdir()) / "aipm_latest_cache")))
                with tempfile.NamedTemporaryFile(suffix=Path(uploaded.name).suffix, delete=False) as stream:
                    stream.write(uploaded.getvalue())
                    source = Path(stream.name)
                try:
                    with st.status("Разбираем ролик", expanded=True) as status:
                        result = run_latest_analysis(
                            source_video=source, output_root=cache, api_key=_api_key() if live_btn else "",
                            vertical=VERTICALS[vertical], allow_live=bool(live_btn), material_kind=material_kind,
                            repeat_count=ANALYSIS_REPEATS,
                            progress=lambda message: status.update(label=(
                                "Проверяем наблюдения по ролику" if message.startswith("Проверяем")
                                else "Анализируем содержание ролика")),
                            evidence_collector=_review_collector())
                        result["source_name"] = uploaded.name
                        st.session_state[result_key] = result
                        status.update(label="Разбор завершён", state="complete", expanded=False)
                except MissingMeasurement:
                    st.warning("Сохранённого анализа этого ролика пока нет. "
                               "Для продолжения нажмите «Проанализировать ролик».")
                except AnalysisBusy as exc:
                    st.warning(str(exc))
                except Exception as exc:
                    st.error("Не удалось завершить анализ. "
                             "Готовые этапы сохранены; повторный запуск продолжит расчёт.")
                finally:
                    source.unlink(missing_ok=True)
        if uploaded is not None and result_key in st.session_state:
            current_sha = hashlib.sha256(uploaded.getvalue()).hexdigest()
            stored_sha = st.session_state[result_key].get("source_sha")
            if stored_sha and stored_sha != current_sha:
                st.info("Ниже показан результат предыдущего ролика. Запустите анализ нового файла.")
            elif st.session_state[result_key].get("vertical") not in (None, VERTICALS[vertical]):
                st.info("Вертикаль изменена. Ниже сохранён результат прежнего выбора; "
                        "пересчитайте его по сохранённым наблюдениям.")

    if result_key in st.session_state:
        st.divider()
        result = st.session_state[result_key]
        if material_kind_for_result(result) != material_kind:
            st.error("Сохранённый результат относится к другому типу материала. Откройте соответствующий раздел.")
            st.stop()
        st.caption(result.get("source_name", result.get("name", result.get("record", ""))))
        present = celebrity_presence(result)
        selected_celebrity = st.checkbox(
            "В ролике есть медийная персона", value=present is True,
            key=prefix + "celebrity_" + str(result.get("source_sha", result.get("record", ""))))
        if selected_celebrity != present:
            result = with_celebrity_review(result, selected_celebrity)
            st.session_state[result_key] = result
        if material_kind == "neuromatics":
            choices = {"Полнота озвучки не подтверждена": "unknown", "Вся речь есть (можно черновую)": "complete",
                       "Есть только часть речи": "partial", "Речи нет": "absent"}
            status = audio_status(result)
            selected = st.selectbox("Озвучка в этом файле", list(choices),
                                    index=list(choices.values()).index(status),
                                    key=prefix + "audio_" + str(result.get("source_sha", "")),
                                    help="Музыка и звуковые эффекты не заменяют реплики и закадровый текст. Если речи нет по замыслу, выберите «Речи нет»: отдельной нормы для таких роликов пока нет.")
            if choices[selected] != status:
                result = dict(result, audio_review=dict(status=choices[selected],
                    source_sha=result.get("source_sha"), origin="user_declared"))
                st.session_state[result_key] = result
        with st.spinner("Готовим объяснение оценок"):
            show_latest_result(result)


if __name__ in {"__main__", "__page__"}:
    main()
