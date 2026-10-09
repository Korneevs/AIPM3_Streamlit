"""Main pretest page for the frozen latest model and evidence-based review."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import tempfile

import streamlit as st

from aipm3.latest_interpretation import VERSION, build_latest_interpretation, public_result
from aipm3.latest_pipeline import MissingMeasurement, run_latest_analysis
from aipm3.latest_runtime import (
    ANALYSIS_REPEATS, NEUROMATICS_ANALYSIS_REPEATS, MATERIAL_LABELS, clean_json,
    material_kind_for_result, validate_cached_result,
)
from aipm3.runtime_resources import AnalysisBusy, AnalysisTimeout
from aipm3.manager_report import report_cards
from aipm3.display_calibration import audio_status
from aipm3.latest_manual_inputs import celebrity_presence, with_celebrity_review
from aipm3.latest_profile_ui import show_summary, show_profile
from aipm3.neuromatics_manager import VERSION as MANAGER_VERSION, FINISHED_VERSION
from aipm3.manager_display import VERSION as DISPLAY_VERSION
from aipm3.latest_uvp import main_idea, target_for_vertical, with_blind_answers, with_uvp
from aipm3.vertical_uvp import GOODS
from aipm3.uvp_ui import show_uvp


VERTICALS = {"Товары": "Goods", "Авто": "Auto", "Работа": "Jobs",
             "Недвижимость": "Realty", "Услуги": "Services", "Путешествия": "Travel"}


@st.cache_data(show_spinner=False)
def _interpret_cached(result: dict, version: str) -> dict:
    return build_latest_interpretation(result, evidence=result.get("independent_evidence", []))


def interpret_result(result: dict) -> dict:
    # Review metadata must not force a second attribution calculation.
    neuro = material_kind_for_result(result) == "neuromatics"
    base = {k: v for k, v in result.items() if k != "manager_review"}
    cache_version = ':'.join([VERSION, MANAGER_VERSION if neuro else FINISHED_VERSION, DISPLAY_VERSION])
    interpretation = _interpret_cached(base, cache_version)
    if result.get("manager_review"):
        from aipm3.interpretation_checker import apply_review
        return apply_review(interpretation, result["manager_review"])
    return interpretation


def check_manager_interpretation(result: dict, api_key: str, cache: Path) -> dict:
    interpretation = interpret_result({k: v for k, v in result.items() if k != "manager_review"})
    draft = interpretation.get("manager_semantic")
    if not draft:
        return result
    from aipm3.interpretation_checker import review_draft
    receipt = review_draft(draft, api_key=api_key, cache_dir=cache)
    return dict(result, manager_review=receipt)


def show_latest_result(result: dict, interpretation: dict | None = None) -> None:
    material_kind = material_kind_for_result(result)
    interpretation = interpretation or interpret_result(result)
    if interpretation.get("audio_note"):
        st.warning(interpretation["audio_note"])
    show_summary(interpretation)
    with st.container(border=True):
        st.markdown("### Основная идея ролика")
        st.write(main_idea(result))
    assessment = result.get("vertical_uvp")
    if assessment and assessment.get("source_sha") == result.get("source_sha"):
        show_uvp(result)
    else:
        with st.container(border=True):
            st.markdown("### Попадание в UVP")
            st.write("Выберите целевую выгоду в боковой панели и нажмите «Проверить UVP».")
    show_profile(interpretation)
    from aipm3.neuromatics_manager import manager_cards
    cards = manager_cards(interpretation)
    st.caption("Оценка помогает сравнивать варианты. Финальное решение принимает Марком. "
               "Предложения по правкам стоит проверить на следующей версии ролика.")
    exported = clean_json(public_result(with_blind_answers(result), interpretation))
    exported["main_idea"] = main_idea(result)
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
    analysis_repeats = NEUROMATICS_ANALYSIS_REPEATS if material_kind == "neuromatics" else ANALYSIS_REPEATS
    result_key = prefix + "result"
    st.title(MATERIAL_LABELS[material_kind])
    st.caption("Что поддерживает оценку ролика, что её ограничивает и какие выводы пока нельзя сделать.")
    mode = st.sidebar.radio("Источник результата", ["Загрузить ролик", "Открыть сохранённый результат"],
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
        with st.sidebar:
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
        stored_vertical = st.session_state.get(result_key, {}).get("vertical")
        default_vertical = list(VERTICALS.values()).index(stored_vertical) if stored_vertical in VERTICALS.values() else 0
        vertical = st.sidebar.selectbox("Вертикаль", list(VERTICALS), index=default_vertical,
                                       key=prefix + "vertical")
        analysis_target = None
        if material_kind == "neuromatics":
            goods = None
            if VERTICALS[vertical] == "Goods":
                goods = st.sidebar.selectbox("Направление товаров", list(GOODS), index=None,
                    placeholder="Ресейл или распродажа", key=prefix + "upload_goods")
            analysis_target = target_for_vertical(VERTICALS[vertical], goods)
            if analysis_target:
                st.sidebar.caption("Целевой UVP: " + analysis_target["label"] + " - " + analysis_target["meaning"])
        else:
            analysis_target = target_for_vertical(VERTICALS[vertical])
        with st.sidebar:
            uploaded = st.file_uploader("Ролик Avito (MP4 / MOV)", type=["mp4", "mov"], key=prefix + "video")
        if uploaded is not None:
            st.video(uploaded)
        live_btn = st.sidebar.button("Проанализировать ролик", type="primary", use_container_width=True,
                               key=prefix + "live")
        st.caption(f"Ролик оценивается {analysis_repeats} {'раза' if analysis_repeats == 3 else 'раз'}; "
                   "итоговые оценки усредняются. "
                   "Анализ и проверка наблюдений могут занять несколько минут. "
                   "Если сервер занят, ролик встанет в очередь и анализ начнётся автоматически.")
        if live_btn:
            if uploaded is None:
                st.warning("Сначала загрузите ролик.")
            elif material_kind == "neuromatics" and analysis_target is None:
                st.warning("Выберите направление товаров для проверки UVP: ресейл или распродажа.")
            elif not _api_key():
                st.error("Новый анализ пока недоступен. Обратитесь к администратору приложения.")
            else:
                cache = Path(os.environ.get("AIPM_LATEST_CACHE_DIR",
                                            str(Path(tempfile.gettempdir()) / "aipm_latest_cache")))
                with tempfile.NamedTemporaryFile(suffix=Path(uploaded.name).suffix, delete=False) as stream:
                    stream.write(uploaded.getvalue())
                    source = Path(stream.name)
                try:
                    with st.status("Разбираем ролик", expanded=True) as status:
                        bar = st.progress(0, text="Ожидаем запуска анализа")
                        measurement_steps = 16 * analysis_repeats
                        total_steps = measurement_steps + 8

                        def progress(message):
                            status.update(label=message)
                            reading = re.match(r"Повтор (\d+)/(\d+): выполнено проверок (\d+)/16", message)
                            step = 0
                            if reading:
                                repeat, _, done = map(int, reading.groups())
                                step = (repeat - 1) * 16 + done
                            elif message.startswith("Повтор "):
                                repeat = int(message.split()[1].split("/")[0])
                                step = (repeat - 1) * 16
                            elif message.startswith("Проверяем наблюдения по видео:"):
                                step = measurement_steps + int(message.rsplit(" ", 1)[1].split("/")[0]) - 1
                            elif message.startswith("Проверяем показанные действие и результат:"):
                                step = measurement_steps + 3 + int(message.rsplit(" ", 1)[1].split("/")[0]) - 1
                            bar.progress(step / total_steps, text=message)

                        result = run_latest_analysis(
                            source_video=source, output_root=cache, api_key=_api_key(),
                            vertical=VERTICALS[vertical], allow_live=True, material_kind=material_kind,
                            repeat_count=analysis_repeats,
                            progress=progress,
                            evidence_collector=_review_collector())
                        result["source_name"] = uploaded.name
                        if celebrity_presence(result) is None:
                            result = with_celebrity_review(result, False)
                        st.session_state[result_key] = result
                        result["main_idea"] = main_idea(result)
                        if analysis_target:
                            status.update(label="Проверяем попадание в UVP")
                            bar.progress((measurement_steps + 6) / total_steps, text="Проверяем попадание в UVP")
                            result = with_uvp(result, analysis_target, _api_key(),
                                              Path(tempfile.gettempdir()) / "aipm3_uvp_cache")
                        status.update(label="Проверяем пояснения к ролику")
                        bar.progress((total_steps - 1) / total_steps, text="Проверяем пояснения к ролику")
                        result = check_manager_interpretation(result, _api_key(), cache)
                        st.session_state[result_key] = result
                        bar.progress(1., text="Разбор готов")
                        status.update(label="Разбор завершён", state="complete", expanded=False)
                except MissingMeasurement:
                    st.warning("Сохранённого анализа этого ролика пока нет. "
                               "Для продолжения нажмите «Проанализировать ролик».")
                except AnalysisBusy as exc:
                    st.warning(str(exc))
                except AnalysisTimeout as exc:
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
                        "для обновления нажмите «Проанализировать ролик».")

    if result_key in st.session_state:
        st.divider()
        result = st.session_state[result_key]
        if material_kind_for_result(result) != material_kind:
            st.error("Сохранённый результат относится к другому типу материала. Откройте соответствующий раздел.")
            st.stop()
        st.caption(result.get("source_name", result.get("name", result.get("record", ""))))
        present = celebrity_presence(result)
        selected_celebrity = st.sidebar.checkbox(
            "В ролике есть медийная персона", value=present is True,
            key=prefix + "celebrity_" + str(result.get("source_sha", result.get("record", ""))))
        if selected_celebrity != present:
            result = with_celebrity_review(result, selected_celebrity)
            st.session_state[result_key] = result
        if material_kind == "neuromatics":
            choices = {"Полнота озвучки не подтверждена": "unknown", "Вся речь есть (можно черновую)": "complete",
                       "Есть только часть речи": "partial", "Речи нет": "absent"}
            status = audio_status(result)
            selected = st.sidebar.selectbox("Озвучка в этом файле", list(choices),
                                    index=list(choices.values()).index(status),
                                    key=prefix + "audio_" + str(result.get("source_sha", "")),
                                    help="Музыка и звуковые эффекты не заменяют реплики и закадровый текст. Если речи нет по замыслу, выберите «Речи нет»: отдельной нормы для таких роликов пока нет.")
            if choices[selected] != status:
                result = dict(result, audio_review=dict(status=choices[selected],
                    source_sha=result.get("source_sha"), origin="user_declared"))
                st.session_state[result_key] = result
        with st.sidebar:
            st.subheader("Проверка UVP")
            # Use the scored vertical, not a newly selected upload's settings.
            scored_vertical = result.get("vertical") or result.get("metadata", {}).get("vertical")
            goods = None
            if scored_vertical == "Goods":
                stored_goods = result.get("vertical_uvp", {}).get("target", {}).get("goods")
                goods = st.selectbox("Направление товаров", list(GOODS),
                                     index=(list(GOODS).index(stored_goods)
                                            if material_kind == "neuromatics" and stored_goods in GOODS else None),
                                     placeholder="Ресейл или распродажа", key=prefix + "goods")
            target = target_for_vertical(scored_vertical, goods) if scored_vertical else None
            if target:
                st.caption("Целевой UVP: " + target["label"] + " - " + target["meaning"])
                current = result.get("vertical_uvp", {})
                if current and current.get("target") != target:
                    st.caption("Ниже сохранена проверка для прежнего выбора.")
                if st.button("Проверить UVP", key=prefix + "uvp"):
                    if not _api_key():
                        st.warning("Для новой проверки UVP нужен доступ к анализу.")
                    else:
                        with st.spinner("Проверяем UVP по сохранённым ответам"):
                            result = with_uvp(result, target, _api_key(),
                                              Path(tempfile.gettempdir()) / "aipm3_uvp_cache")
                        st.session_state[result_key] = result
            else:
                st.caption("Выберите направление товаров для проверки UVP." if scored_vertical == "Goods"
                           else "В сохранённом разборе не указана вертикаль для проверки UVP.")
        with st.spinner("Готовим объяснение оценок"):
            show_latest_result(result)


if __name__ in {"__main__", "__page__"}:
    main()
