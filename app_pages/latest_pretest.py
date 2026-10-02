"""Main pretest page for the frozen latest model and evidence-based review."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile

import plotly.graph_objects as go
import streamlit as st

from aipm3.latest_interpretation import VERSION, build_latest_interpretation, public_result
from aipm3.latest_pipeline import MissingMeasurement, run_latest_analysis
from aipm3.latest_runtime import SCORING_VERSION, clean_json, validate_cached_result
from aipm3.runtime_resources import AnalysisBusy


VERTICALS = {"Товары": "Goods", "Авто": "Auto", "Работа": "Jobs",
             "Недвижимость": "Realty", "Услуги": "Services", "Путешествия": "Travel"}


@st.cache_data(show_spinner=False)
def interpret_result(result: dict, version: str = VERSION) -> dict:
    return build_latest_interpretation(result, evidence=result.get("independent_evidence", []))


def comparison_figure(cards: list[dict]) -> go.Figure:
    colors = {"Выше типичного уровня": "#27836D", "Типичный уровень": "#5479B9",
              "Ниже типичного уровня": "#C9862F"}
    figure = go.Figure(go.Bar(
        x=[c["index"] for c in cards], y=[c["title"] for c in cards], orientation="h",
        marker_color=[colors.get(c["level"], "#5479B9") for c in cards],
        text=[f'{c["index"]:.0f}' for c in cards], textposition="outside",
        hovertemplate="%{y}: %{x:.1f}<extra></extra>",
    ))
    figure.add_vline(x=100, line_dash="dot", line_color="#8793A4")
    figure.update_layout(height=240, margin=dict(l=5, r=40, t=20, b=20),
                         xaxis_title="Индекс · 100 = средняя оценка исторических роликов",
                         yaxis=dict(autorange="reversed"), showlegend=False,
                         paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return figure


def _show_driver(driver: dict) -> None:
    st.markdown(f'**{driver["label"]}**')
    if 'index_points' in driver:
        st.caption(f'Вклад в оценку: {driver["index_points"]:+.1f} пункта индекса'.replace('.', ','))
    support = driver["evidence"]
    st.write(support["observation"])
    episodes = support.get("episodes", [])
    if episodes:
        st.caption("В ролике: " + "; ".join(
            f'{float(e["start"]):g}–{float(e["end"]):g} с' for e in episodes))
    if driver.get("interpretation_kind") == "association_only":
        st.caption("Особенность модели, не рекомендация")
    if driver.get("why") and driver.get('interpretation_kind')=='association_only':
        st.write(driver["why"])


def _show_partial_explanation(card: dict) -> None:
    unresolved = sorted(card.get("unresolved_drivers", []),
                        key=lambda driver: -abs(driver["index_points"]))
    if card.get("explanation_status") == "partial":
        message = "**Разбор причин частичный.** "
        if unresolved:
            largest = unresolved[0]
            points = f'{largest["index_points"]:+.1f}'.replace("-", "−").replace(".", ",")
            message += (f'Наибольший вклад, который требует проверки: «{largest["label"]}», '
                        f"{points} пункта индекса. Эту причину пока нельзя уверенно объяснить по видео.")
        else:
            message += "Часть существенных вкладов пока нельзя объяснить подтверждёнными наблюдениями по видео."
        st.warning(message)
    if unresolved:
        with st.expander("Причины в расчёте, которые требуют проверки"):
            st.caption("Вклады ниже относятся к исходным признакам модели. "
                       "Они не подтверждают наличие свойства в ролике и не показывают эффект правки.")
            for driver in unresolved:
                points = f'{driver["index_points"]:+.1f}'.replace("-", "−").replace(".", ",")
                st.markdown(f'**{driver["label"]} · {points} пункта индекса**')
                reason = driver.get("support_reason") or driver.get("evidence", {}).get("observation")
                if reason:
                    st.write(reason)
                measurement = driver.get("model_measurement", driver.get("value"))
                if measurement is not None:
                    value = f"{measurement:g}" if isinstance(measurement, (float, int)) else str(measurement)
                    st.caption("Значение в исходном разборе: " + value)


def show_latest_result(result: dict, interpretation: dict | None = None) -> None:
    interpretation = interpretation or interpret_result(result)
    overall=interpretation.get('overall')
    if overall:
        st.subheader('Общая оценка · AIPM 3.0')
        st.metric(overall['title'],f'{overall["index"]:.0f}')
        st.caption(overall['level']+' · Среднее по 10 полным прогонам')
    st.subheader("Результат по трём компонентам")
    st.caption(interpretation["scale_note"])
    for column, card in zip(st.columns(3), interpretation["cards"]):
        with column:
            st.metric(card["title"], f'{card["index"]:.0f}')
            st.caption(card["level"])
    with st.expander("Сравнение и разброс повторных разборов"):
        st.plotly_chart(comparison_figure(interpretation["cards"]), use_container_width=True)
        for card in interpretation["cards"]:
            low, high = card["repeat_index_range"]
            st.write(f'{card["title"]}: {low:.0f}–{high:.0f}')
        st.caption(interpretation["repeat_note"])

    for card in interpretation["cards"]:
        with st.container(border=True):
            st.markdown(f'### {card["title"]}')
            _show_partial_explanation(card)
            strengths, limitations = st.columns(2)
            with strengths:
                st.markdown("**Что модель учла в плюс**")
                for driver in card["strengths"]:
                    _show_driver(driver)
                if not card["strengths"]:
                    st.caption("Нет достаточно устойчивых и подтверждённых наблюдений.")
            with limitations:
                st.markdown("**Что модель учла в минус**")
                for driver in card["limitations"]:
                    _show_driver(driver)
                if not card["limitations"]:
                    st.caption("Нет достаточно устойчивых и подтверждённых наблюдений.")
            checks = [d for d in [*card["strengths"], *card["limitations"]] if d.get("check")]
            if checks:
                st.markdown("**Что проверить в следующей версии**")
                for driver in checks:
                    st.write(driver["check"])
            if card["unresolved"]:
                st.caption("Требуют проверки по видео: " + "; ".join(card["unresolved"]) + ".")
    st.caption(interpretation["interpretation_note"])
    if not interpretation.get("evidence_runs"):
        st.info("Независимая проверка сюжетных наблюдений ещё не выполнена. "
                "Баллы рассчитаны; неподтверждённые причины не превращены в рекомендации.")
    exported = clean_json(public_result(result,interpretation))
    with st.expander("Версия расчёта"):
        st.code(SCORING_VERSION, language=None)
        st.caption("Замороженная модель от 1 октября 2026: 9 / 9 / 7 признаков. "
                   "Итог Q — среднее произведение трёх оценок по десяти повторам. "
                   "Независимая проверка качества модели на отложенной выборке ещё не завершена.")
        st.write({"Q": result["scores"]["Q"], "OPM": result["scores"]["OPM"]})
    st.download_button("Скачать результат и интерпретацию", data=json.dumps(
        exported, ensure_ascii=False, indent=2, allow_nan=False),
        file_name="aipm_20261001_result.json", mime="application/json")


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


def main() -> None:
    st.title("Видео-претест · AIPM 3.0")
    st.caption("Заметность, считываемость и запоминаемость — с наблюдениями для обсуждения ролика.")
    mode = st.radio("Источник результата", ["Загрузить ролик", "Открыть сохранённый результат"],
                    horizontal=True)
    preset = os.environ.get("AIPM_LATEST_RESULT_JSON")
    if preset and st.session_state.get("latest_preset") != preset:
        try:
            st.session_state["latest_result"] = validate_cached_result(json.loads(Path(preset).read_text()))
            st.session_state["latest_preset"] = preset
        except (ValueError, KeyError, OSError) as exc:
            st.error(str(exc))

    if mode == "Открыть сохранённый результат":
        saved = st.file_uploader("Результат анализа (JSON)", type=["json"], key="latest_saved")
        if saved is not None:
            digest = hashlib.sha256(saved.getvalue()).hexdigest()
            if st.session_state.get("latest_saved_sha") != digest:
                try:
                    st.session_state["latest_result"] = validate_cached_result(json.loads(saved.getvalue()))
                    st.session_state["latest_saved_sha"] = digest
                except (ValueError, KeyError, TypeError) as exc:
                    st.error(str(exc))
                    st.stop()
        st.caption("Открытие сохранённого результата не отправляет запросы к AI.")
    else:
        vertical = st.selectbox("Вертикаль", list(VERTICALS))
        uploaded = st.file_uploader("Ролик Avito (MP4 / MOV)", type=["mp4", "mov"], key="latest_video")
        if uploaded is not None:
            st.video(uploaded)
        cached, live = st.columns(2)
        cached_btn = cached.button("Рассчитать по сохранённым наблюдениям", use_container_width=True)
        live_btn = live.button("Запустить AI-анализ", type="primary", use_container_width=True)
        st.caption("AI-анализ отправляет видео на проверку: 10 повторов разбора "
                   "и отдельные проверки наблюдений. Готовые этапы используются повторно.")
        if cached_btn or live_btn:
            if uploaded is None:
                st.warning("Сначала загрузите ролик.")
            elif live_btn and not _api_key():
                st.error("Для нового анализа нужен настроенный ключ VSELLM_API_KEY.")
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
                            vertical=VERTICALS[vertical], allow_live=bool(live_btn),
                            progress=lambda message: status.update(label=message),
                            evidence_collector=_review_collector())
                        result["source_name"] = uploaded.name
                        st.session_state["latest_result"] = result
                        status.update(label="Разбор завершён", state="complete", expanded=False)
                except MissingMeasurement:
                    st.warning("Для этого ролика не хватает сохранённых наблюдений. "
                               "Новые запросы не отправлялись. Для продолжения нажмите «Запустить AI-анализ».")
                except AnalysisBusy as exc:
                    st.warning(str(exc))
                except Exception as exc:
                    st.error(f"Не удалось завершить анализ ({type(exc).__name__}). "
                             "Готовые этапы сохранены; повторный запуск продолжит расчёт.")
                finally:
                    source.unlink(missing_ok=True)
        if uploaded is not None and "latest_result" in st.session_state:
            current_sha = hashlib.sha256(uploaded.getvalue()).hexdigest()
            stored_sha = st.session_state["latest_result"].get("source_sha")
            if stored_sha and stored_sha != current_sha:
                st.info("Ниже показан результат предыдущего ролика. Запустите анализ нового файла.")
            elif st.session_state["latest_result"].get("vertical") not in (None, VERTICALS[vertical]):
                st.info("Вертикаль изменена. Ниже сохранён результат прежнего выбора; "
                        "пересчитайте его по сохранённым наблюдениям.")

    if "latest_result" in st.session_state:
        st.divider()
        result = st.session_state["latest_result"]
        st.caption(result.get("source_name", result.get("name", result.get("record", ""))))
        with st.spinner("Готовим объяснение оценок"):
            show_latest_result(result)


if __name__ in {"__main__", "__page__"}:
    main()
