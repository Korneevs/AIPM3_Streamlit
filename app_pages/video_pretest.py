from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from artifacts import artifact_path
from aipm3.models import FrozenModels, level_from_percentile, load_frozen_models
from aipm3.pipeline import run_analysis


@st.cache_resource(show_spinner=False)
def load_models() -> FrozenModels:
    return load_frozen_models(
        artifact_path("aipm1_model.cbm"),
        artifact_path("aipm2_model.cbm"),
        artifact_path("message_delivery_model_bundle.joblib"),
    )


def component_figure(result: dict) -> go.Figure:
    labels = ["Заметность", "Запоминаемость", "Считываемость"]
    values = [
        result["aipm1"]["percentile"],
        result["aipm2"]["percentile"],
        result["message_delivery"]["percentile"],
    ]
    colors = ["#1EA7FD" if value >= 67 else "#FFB020" if value >= 33 else "#E44D61" for value in values]
    figure = go.Figure(go.Bar(
        x=values,
        y=labels,
        orientation="h",
        marker_color=colors,
        text=[f"{value:.0f}/100" for value in values],
        textposition="inside",
        hovertemplate="%{y}: %{x:.0f}/100<extra></extra>",
    ))
    figure.add_vline(x=33, line_dash="dot", line_color="#B8C0CC")
    figure.add_vline(x=67, line_dash="dot", line_color="#B8C0CC")
    figure.update_layout(
        height=285,
        margin=dict(l=10, r=10, t=15, b=15),
        xaxis=dict(range=[0, 100], title="Позиция относительно референсных роликов"),
        yaxis=dict(autorange="reversed", title=""),
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return figure


def group_figure(rows: list[dict]) -> go.Figure:
    frame = pd.DataFrame(rows)
    frame = frame.loc[frame["Группа"].ne("Контекст бренда")].copy()
    frame["label"] = frame["Компонент"] + " · " + frame["Группа"]
    frame = frame.sort_values("Локальное влияние, %")
    colors = frame["Локальное влияние, %"].map(
        lambda value: "#27AE60" if value > 0 else "#E44D61"
    )
    figure = go.Figure(go.Bar(
        x=frame["Локальное влияние, %"],
        y=frame["label"],
        orientation="h",
        marker_color=colors,
        hovertemplate="%{y}<br>Локальное влияние: %{x:.1f}%<extra></extra>",
    ))
    figure.add_vline(x=0, line_color="#6B7280", line_width=1)
    figure.update_layout(
        height=max(380, 37 * len(frame)),
        margin=dict(l=10, r=10, t=20, b=15),
        xaxis_title="Доля локального влияния внутри компонента, %",
        yaxis_title="",
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return figure


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
        yaxis=dict(title="Доля валидно считавших идею", tickformat=".0%", range=[0, 1]),
        legend=dict(orientation="h", y=1.12),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return figure


def show_result(result: dict) -> None:
    aipm3 = result["aipm3"]
    if aipm3["level"] == 2:
        st.success(f"**AIPM 3.0: {aipm3['label']}**")
    elif aipm3["level"] == 1:
        st.warning(f"**AIPM 3.0: {aipm3['label']}**")
    else:
        st.error(f"**AIPM 3.0: {aipm3['label']}**")

    col_total, col_aipm1, col_aipm2, col_md = st.columns(4)
    col_total.metric(
        "AIPM 3.0",
        f"{aipm3['index_100']:.0f}",
        help="Нормированный индекс: 100 — средний уровень референсной выборки.",
    )
    components = [
        (col_aipm1, "Заметность · AIPM 1.0", result["aipm1"]["percentile"]),
        (col_aipm2, "Запоминаемость · AIPM 2.0", result["aipm2"]["percentile"]),
        (col_md, "Считываемость · MD", result["message_delivery"]["percentile"]),
    ]
    for column, label, value in components:
        column.metric(label, f"{value:.0f}/100", level_from_percentile(value))

    st.plotly_chart(component_figure(result), use_container_width=True)

    st.subheader("Как читать результат")
    st.info(result["interpretation"]["summary"])
    st.markdown(f"**Главная идея, считанная синтетическими респондентами:** {result['main_idea']}")

    good_col, bad_col = st.columns(2)
    with good_col:
        st.markdown("### Что в ролике работает")
        for item in result["interpretation"]["strengths"]:
            with st.container(border=True):
                st.markdown(f"**{item['title']}**")
    with bad_col:
        st.markdown("### Что ограничивает результат")
        for item in result["interpretation"]["limits"]:
            with st.container(border=True):
                st.markdown(f"**{item['title']}**")

    st.subheader("Разбор по логическим группам признаков")
    st.caption(
        "Зелёное — группа поддерживает компонент, красное — ограничивает. "
        "Проценты сравнимы только внутри одного компонента."
    )
    st.plotly_chart(
        group_figure(result["interpretation"]["group_rows"]),
        use_container_width=True,
    )

    with st.expander("Устойчивость идеи при неполном просмотре"):
        st.plotly_chart(recovery_figure(result["recovery_curve"]), use_container_width=True)
        st.caption(
            "Две линии — независимые наборы фрагментов ролика. Чем выше линии и чем "
            "меньше расстояние между ними, тем устойчивее считывается главная идея."
        )

    downloadable = {
        key: value for key, value in result.items()
        if key not in {"objective_runs", "transcripts", "prepared_video"}
    }
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
    frozen_models = load_models()

with st.sidebar:
    st.header("Настройки")
    uploaded_file = st.file_uploader("Загрузите ролик (MP4 / MOV)", type=["mp4", "mov"])
    st.caption("AIPM 1.0: 3 просмотра. AIPM 2.0: 2 просмотра. Message Delivery: 30 респондентов.")
    analyze_btn = st.button("Начать анализ", type="primary", use_container_width=True)

if uploaded_file is not None:
    st.video(uploaded_file)

if analyze_btn:
    if uploaded_file is None:
        st.warning("Сначала загрузите ролик.")
    else:
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
            st.session_state["aipm3_result"] = result
        except Exception as exc:
            st.exception(exc)
        finally:
            if source_path.exists():
                os.remove(source_path)

if "aipm3_result" in st.session_state:
    st.divider()
    show_result(st.session_state["aipm3_result"])
