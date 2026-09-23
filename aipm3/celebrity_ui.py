"""Manager-facing manual participation, without exposing policy mechanics."""
from html import escape

import streamlit as st

from .manual_celebrity import CELEBRITIES, VERSION


def show_celebrity(result: dict) -> None:
    adjustment = result.get("celebrity_adjustment", {})
    if adjustment.get("version") != VERSION:
        return
    present = adjustment["present"]
    with st.container(border=True):
        description, effects = st.columns([1.5, 1], gap="large")
        with description:
            st.markdown("#### Участие селебрити")
            st.write(CELEBRITIES[adjustment["selection"]] if present else "Без селебрити из списка.")
            st.caption("Участие указано вручную")
        with effects:
            status, color = ("Повышает", "#137547") if present else ("Не влияет", "#667085")
            st.markdown(
                f'<div style="color:{color};font-weight:600;margin:12px 0 5px;">'
                f'{escape(status)} · запоминаемость</div>'
                '<div style="font-size:14px;">Вес в итоговой запоминаемости: '
                f'<strong>{adjustment["final_share_percent"]:.0f}%</strong></div>',
                unsafe_allow_html=True,
            )
