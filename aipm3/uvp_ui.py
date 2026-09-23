"""Presentation for the independent, post-scoring vertical UVP check."""
import streamlit as st

from .vertical_uvp import STATUS_LABELS


def show_uvp(result):
    assessment = result.get("vertical_uvp")
    if not assessment:
        return
    target = assessment["target"]
    with st.container(border=True):
        st.markdown("### Попадание в UVP")
        context = target["vertical"] + (" · " + target["goods"] if target.get("goods") else "")
        if target.get("period"):
            context += " · " + target["period"]
        st.write(context + " → **" + target["label"] + "**")
        st.write(target["meaning"])
        status = assessment.get("status", "error")
        label = STATUS_LABELS.get(status, "Проверка UVP не завершилась; оценки AIPM сохранены")
        {"matched": st.success, "partial": st.warning, "absent": st.error}.get(status, st.info)(label)
        if status in {"error", "insufficient"}:
            return
        n, counts = assessment["total"], assessment["counts"]
        st.write(f'Целевая выгода выражена в **{counts["matched"]} из {n}** автоматических пересказов; '
                 f'частично — в **{counts["partial"]}**; не выражена — в **{counts["absent"]}**.'
                 + (f' Противоположный смысл — в **{counts["contradicted"]}**.' if counts["contradicted"] else ""))
        st.caption("Проверка по ответам модели, полученным без подсказки UVP. Это не опрос живых зрителей и не часть оценки AIPM.")
        categories = [("matched", "Целевая выгода"), ("partial", "Только часть смысла"),
                      ("absent", "Другой акцент"), ("contradicted", "Противоположный смысл")]
        # Show evidence for both the positive and negative side when present.
        for key, title in categories:
            example = next((r for r in assessment["answers"] if r["status"] == key), None)
            if example:
                st.markdown("**" + title + ":**")
                st.write('«' + (example["quote"] or example["answer"]) + '»')
        with st.expander("Критерий UVP и правило вывода"):
            st.write(target["criterion"])
            st.write(target["exclude"])
            st.write("«Попали» — больше половины пересказов передают целевую выгоду полностью. "
                     "«Частично» — не меньше половины передают её полностью или частично. "
                     "Иначе — «Не попали». Это правило смысловой проверки, не норматив эффективности рекламы.")
