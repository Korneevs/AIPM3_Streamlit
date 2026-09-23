from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

from artifacts import artifact_path
from aipm3.models import SCORING_VERSION, FrozenModels, load_frozen_models
from aipm3.pipeline import run_analysis
from aipm3.profile_ui import show_feature_profile
from aipm3.feature_profile import build_profile
from aipm3.result_export import export_result
from aipm3.summary_ui import show_metric_summary
from aipm3.marketing_profile import build_marketing_profile
from aipm3.runtime_resources import AnalysisBusy
from aipm3.vertical_uvp import VERTICALS, GOODS, make_target, evaluate_uvp
from aipm3.uvp_ui import show_uvp
from aipm3.manual_celebrity import CELEBRITIES, apply_celebrity, selected_celebrity


def with_uvp(result: dict, target: dict, api_key: str) -> dict:
    try:
        assessment = evaluate_uvp(result, target, api_key,
                                  Path(tempfile.gettempdir()) / "aipm3_uvp_cache")
    except Exception:
        # A failed optional diagnostic must never discard finished model scores.
        assessment = {"target": dict(target), "status": "error", "source_sha": result.get("source_sha")}
    return {**result, "vertical_uvp": assessment}


@st.cache_resource(show_spinner=False)
def load_models(scoring_version: str) -> FrozenModels:
    return load_frozen_models(
        artifact_path("aipm1_model.cbm"),
        artifact_path("aipm2_model.cbm"),
        artifact_path("message_delivery_model_bundle.joblib"),
    )


def recovery_figure(curve: dict[str, float]) -> go.Figure:
    fractions = [25, 50, 75, 100]
    full = float(curve["full"])
    mask1 = [curve["nested_25_m1"], curve["nested_50_m1"], curve["nested_75_m1"], full]
    mask2 = [curve["nested_25_m2"], curve["nested_50_m2"], curve["nested_75_m2"], full]
    figure = go.Figure()
    figure.add_trace(go.Scatter(
        x=fractions, y=mask1, mode="lines+markers", name="Recovery-маска 1",
        line=dict(color="#1EA7FD", width=3),
    ))
    figure.add_trace(go.Scatter(
        x=fractions, y=mask2, mode="lines+markers", name="Recovery-маска 2",
        line=dict(color="#7D5FFF", width=3),
    ))
    figure.update_layout(
        height=330,
        margin=dict(l=10, r=10, t=20, b=15),
        xaxis=dict(title="Доля доступного ролика, %", range=[20, 105]),
        yaxis=dict(title="Содержательные ответы (сглаженная доля)", tickformat=".0%", range=[0, 1]),
        legend=dict(orientation="h", y=1.12),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return figure


def show_result(result: dict, api_key: str = "", video_bytes: bytes | None = None, suffix: str = ".mp4") -> None:
    show_metric_summary(result)
    main_idea = str(result.get("main_idea") or "").strip()
    with st.container(border=True):
        st.markdown("### Основная идея ролика")
        st.write(main_idea or "Не удалось однозначно определить основную идею.")

    show_uvp(result)
    show_feature_profile(result)
    downloadable = export_result(result)
    downloadable['feature_profile'] = build_profile(result)
    downloadable['marketing_profile'] = build_marketing_profile(result)
    st.download_button(
        "Скачать результат JSON",
        data=json.dumps(downloadable, ensure_ascii=False, indent=2),
        file_name="aipm3_result.json",
        mime="application/json",
    )


st.title("🎬 AI-Pretest MesSage · AIPM 3.0")
st.caption("Оценка заметности, запоминаемости и считываемости рекламного ролика")

try:
    api_key = str(st.secrets["VSELLM_API_KEY"])
except Exception:
    st.error("Не найден VSELLM_API_KEY в Streamlit Secrets.")
    st.stop()

with st.spinner("Загрузка замороженных моделей..."):
    frozen_models = load_models(SCORING_VERSION)

with st.sidebar:
    st.header("Настройки")
    vertical = st.selectbox("Вертикаль", list(VERTICALS), index=None, placeholder="Выберите вертикаль")
    goods = None
    if vertical == "Товары":
        goods = st.selectbox("Направление товаров", list(GOODS), index=None, placeholder="Ресейл или распродажа")
    target = make_target(vertical, goods) if vertical and (vertical != "Товары" or goods) else None
    if target:
        st.caption("Целевой UVP: " + target["label"] + " — " + target["meaning"]
                   + (" · " + target["period"] if target["period"] else ""))
    celebrity = st.selectbox(
        "Селебрити в ролике", list(CELEBRITIES), format_func=CELEBRITIES.get,
        key="selected_celebrity", help="Укажите участника вручную. Если никого из списка нет, выберите «Нет».",
    )
    uploaded_file = st.file_uploader("Загрузите ролик (MP4 / MOV)", type=["mp4", "mov"])
    st.caption("AIPM 1.0: 3 просмотра. AIPM 2.0: 2 просмотра. Message Delivery: 30 респондентов.")
    analyze_btn = st.button("Начать анализ", type="primary", use_container_width=True)

if uploaded_file is not None and not analyze_btn:
    st.video(uploaded_file)

if analyze_btn:
    if uploaded_file is None:
        st.warning("Сначала загрузите ролик.")
    elif target is None:
        st.warning("Выберите вертикаль; для товаров также укажите ресейл или распродажу.")
    else:
        st.session_state.pop("aipm3_result", None)
        suffix = Path(uploaded_file.name).suffix.lower() or ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as stream:
            with uploaded_file.getbuffer() as buffer:
                stream.write(buffer)
            source_path = Path(stream.name)
        try:
            with st.status("Анализируем ролик", expanded=True) as status:
                result = run_analysis(
                    source_video=source_path,
                    output_root=Path(tempfile.gettempdir()) / "aipm3_analysis_cache",
                    api_key=api_key,
                    models=frozen_models,
                    progress=lambda message: status.update(label=message),
                )
                result = apply_celebrity(result, celebrity)
                st.session_state["aipm3_result"] = result
                status.update(label="Проверяем попадание в UVP выбранной вертикали")
                result = with_uvp(result, target, api_key)
                status.update(label="Анализ завершён", state="complete", expanded=False)
            st.session_state["aipm3_result"] = result
        except AnalysisBusy as exc:
            st.warning(str(exc))
        except Exception as exc:
            st.exception(exc)
        finally:
            if source_path.exists():
                os.remove(source_path)

if "aipm3_result" in st.session_state:
    st.divider()
    if st.session_state["aipm3_result"].get("scoring_version") == SCORING_VERSION:
        stored = st.session_state["aipm3_result"]
        current_sha = None
        if uploaded_file is not None:
            with uploaded_file.getbuffer() as buffer:
                current_sha = hashlib.sha256(buffer).hexdigest()
        if current_sha is not None and current_sha != stored.get("source_sha"):
            st.warning("Загружен другой файл. Нажмите «Начать анализ»: прежние оценки относятся к предыдущему ролику.")
        else:
            if celebrity != selected_celebrity(stored):
                st.info("Выбор селебрити изменён. Ниже сохранена оценка для прежнего выбора.")
                if st.button("Применить выбор селебрити"):
                    stored = apply_celebrity(stored, celebrity)
                    st.session_state["aipm3_result"] = stored
                    st.rerun()
            current_assessment = stored.get("vertical_uvp", {})
            if target and (current_assessment.get("target") != target or current_assessment.get("status") == "error"):
                if current_assessment and current_assessment.get("target") != target:
                    st.info("Выбрана другая вертикаль или направление. Ниже сохранена проверка для прежнего выбора.")
                if st.button("Проверить UVP для выбранной вертикали"):
                    with st.spinner("Проверяем UVP по сохранённым ответам"):
                        stored = with_uvp(stored, target, api_key)
                    st.session_state["aipm3_result"] = stored
            show_result(stored)
    else:
        st.warning("Обновлена нормировка AIPM 3.0. Запустите анализ заново: прежний результат относится к старой версии расчёта.")
