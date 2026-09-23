"""Display frozen scores relative to their reference means, without rescoring."""
from __future__ import annotations

from html import escape

import streamlit as st

from .models import AIPM2_REFERENCE, MD_REFERENCE, aipm2_reference_index, percentile_index


# AIPM1 uses average ranks over all 48 reference rows, whose mean index is 1.
# AIPM2 has 47 observed rows but retains the historical denominator of 49:
# its actual mean index is 97/98, not 1. Normalization here is display-only.
REFERENCE_MEANS = {
    "aipm1": 1.0,
    "aipm2": sum(aipm2_reference_index(value) for value in AIPM2_REFERENCE) / len(AIPM2_REFERENCE),
    "message_delivery": sum(percentile_index(value, MD_REFERENCE, 49.0) for value in MD_REFERENCE) / len(MD_REFERENCE),
}
STYLES = {
    0: ("Ниже нормы", "#B42332", "#FFF0F1", "#F5C2C7"),
    1: ("В норме", "#856000", "#FFF8DB", "#EAD58B"),
    2: ("Выше нормы", "#137547", "#EAF7EF", "#B6DFC7"),
}


def metric_summaries(result: dict) -> list[dict]:
    total = result["aipm3"]
    cards = [{"label": "AIPM 3.0", "delta": (float(total["index"]) - 1.0) * 100.0,
              "level": int(total["level"])}]
    for component, label in [
        ("aipm1", "Заметность · AIPM 1.0"),
        ("aipm2", "Запоминаемость · AIPM 2.0"),
        ("message_delivery", "Считываемость · MD"),
    ]:
        score = result[component]
        percentile = float(score["percentile"])
        cards.append({
            "label": label,
            "delta": (float(score["reference_index"]) / REFERENCE_MEANS[component] - 1.0) * 100.0,
            "level": 0 if percentile < 33.0 else 1 if percentile < 67.0 else 2,
        })
    return cards


def show_metric_summary(result: dict) -> None:
    for column, card in zip(st.columns(4), metric_summaries(result)):
        status, color, background, border = STYLES[card["level"]]
        rounded = round(card["delta"])
        value = f"{rounded:+d}%" if rounded else "0%"
        value = value.replace("-", "−")
        column.markdown(
            f'<div style="background:{background};border:1px solid {border};'
            f'border-radius:10px;padding:18px 16px;color:{color};">'
            f'<div style="font-size:14px;min-height:42px;color:#343745;">{escape(card["label"])}</div>'
            f'<div style="font-size:36px;line-height:1.2;font-weight:700;">{value}</div>'
            '<div style="font-size:13px;margin-top:4px;">к среднему</div>'
            f'<div style="font-size:15px;font-weight:600;margin-top:16px;">{status}</div>'
            '</div>',
            unsafe_allow_html=True,
        )
    st.caption(
        "0% — средний индекс по референсным роликам. Цвет показывает положение "
        "относительно диапазона нормы: небольшое отклонение от среднего может быть в норме."
    )
