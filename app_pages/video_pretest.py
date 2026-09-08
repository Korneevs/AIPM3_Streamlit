from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

import streamlit as st

from artifacts import artifact_path
from aipm3.models import SCORING_VERSION, FrozenModels, load_frozen_models
from aipm3.pipeline import run_analysis
from aipm3.objective_features import PROTOCOL_VERSION
from aipm3.result_ui import show_manager_result, show_session_comparison
from aipm3.result_history import remember_result
from aipm3.result_export import export_result


@st.cache_resource(show_spinner=False)
def load_models(scoring_version: str) -> FrozenModels:
    return load_frozen_models(
        artifact_path("aipm1_model.cbm"),
        artifact_path("aipm2_model.cbm"),
        artifact_path("message_delivery_model_bundle.joblib"),
    )


def show_result(result: dict, source_name: str = "Ролик") -> None:
    explanation, profile = show_manager_result(result)
    downloadable = export_result(result)
    downloadable["feature_profile"] = profile
    downloadable["manager_explanation"] = explanation
    downloadable["source_name"] = Path(source_name).name
    st.download_button(
        "Сохранить оценки и объяснение · JSON",
        data=json.dumps(downloadable, ensure_ascii=False, indent=2),
        file_name=f"{Path(source_name).stem}_aipm3_result.json",
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
    uploaded_file = st.file_uploader("Загрузите ролик (MP4 / MOV)", type=["mp4", "mov"])
    st.caption("Анализ выполняют модели ИИ. Ответы о посыле синтетические; опрос людей не проводится.")
    analyze_btn = st.button("Начать анализ", type="primary", use_container_width=True)

if uploaded_file is not None:
    with st.expander(f"Загруженный ролик: {uploaded_file.name}", expanded="aipm3_result" not in st.session_state):
        st.video(uploaded_file)

if analyze_btn:
    if uploaded_file is None:
        st.warning("Сначала загрузите ролик.")
    else:
        st.session_state.pop("aipm3_result", None)
        suffix = Path(uploaded_file.name).suffix.lower() or ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as stream:
            stream.write(uploaded_file.getvalue())
            source_path = Path(stream.name)
        try:
            with st.status("Анализируем ролик", expanded=True) as status:
                st.write("Подготавливаем видео и запускаем компоненты AIPM 3.0 параллельно.")
                result = run_analysis(
                    source_video=source_path,
                    output_root=Path(tempfile.gettempdir()) / "aipm3_analysis_cache",
                    api_key=api_key,
                    models=frozen_models,
                )
                status.update(label="Анализ завершён", state="complete", expanded=False)
            result["source_name"] = uploaded_file.name
            st.session_state["aipm3_result"] = result
        except Exception as exc:
            st.exception(exc)
        finally:
            if source_path.exists():
                os.remove(source_path)

if "aipm3_result" in st.session_state:
    st.divider()
    if st.session_state["aipm3_result"].get("scoring_version") == SCORING_VERSION:
        stored = st.session_state["aipm3_result"]
        video_bytes = uploaded_file.getvalue() if uploaded_file is not None else None
        suffix = Path(uploaded_file.name).suffix.lower() if uploaded_file is not None else ".mp4"
        if video_bytes is not None and hashlib.sha256(video_bytes).hexdigest() != stored.get("source_sha"):
            st.warning("Загружен другой файл. Нажмите «Начать анализ»: прежние оценки относятся к предыдущему ролику.")
        else:
            source_name = stored.get("source_name") or (
                uploaded_file.name if uploaded_file is not None else "Ролик"
            )
            st.session_state["aipm3_history"] = remember_result(
                st.session_state.get("aipm3_history", []), stored, source_name,
            )
            show_result(stored, source_name)
    else:
        st.warning("Обновлена нормировка AIPM 3.0. Запустите анализ заново: прежний результат относится к старой версии расчёта.")

show_session_comparison(
    st.session_state.get("aipm3_history", []),
    scoring_version=SCORING_VERSION,
    protocol_version=PROTOCOL_VERSION,
)
