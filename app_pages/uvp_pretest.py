"""Interactive UVP pretest for a new clip."""
from __future__ import annotations

from collections import defaultdict
import hashlib
from html import escape
import logging
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import streamlit as st

from aipm3.latest_contracts import BASE_URL
from aipm3.runtime_resources import AnalysisBusy, AnalysisTimeout, analysis_deadline, analysis_slot
from aipm3.uvp_catalog import VERTICALS, questions_for
from aipm3.uvp_media import prepare_evidence
from aipm3.uvp_scoring import score_video


LABELS = {
    3: "Сильная поддержка", 2: "Средняя поддержка", 1: "Слабая поддержка",
    0: "Нет устойчивого сигнала", -1: "Слабый обратный сигнал",
    -2: "Средний обратный сигнал", -3: "Сильный обратный сигнал",
}
COLORS = {
    3: ("#D8F2E4", "#105638"), 2: ("#E8EFE8", "#314F39"),
    1: ("#F7EED0", "#735416"), 0: ("#EDF0EF", "#42524C"),
}
CLARITY = ("Смысл не считывается", "Возможны разные прочтения", "Смысл понятен", "Смысл ясен")
LOGGER = logging.getLogger(__name__)


def api_key() -> str:
    try:
        return st.secrets.get("VSELLM_API_KEY") or os.environ.get("VSELLM_API_KEY", "")
    except Exception:
        return os.environ.get("VSELLM_API_KEY", "")


def analyze(uploaded, vertical, key, queued, progress):
    with analysis_slot(wait=True, on_queue=queued), analysis_deadline(), TemporaryDirectory(prefix="uvp-") as tmp:
        source = Path(tmp) / ("creative" + Path(uploaded.name).suffix.lower())
        with uploaded.getbuffer() as buf, source.open("wb") as dest:
            dest.write(buf)
        progress("Смотрим ролик и разбираем речь")
        evidence = prepare_evidence(source, api_base=BASE_URL, api_key=key)
        return score_video(vertical=vertical, questions=questions_for(vertical), evidence=evidence,
                           api_base=BASE_URL, api_key=key, on_progress=progress)


def show_result(result):
    st.subheader("Основная идея ролика")
    st.html(f"<p>{escape(result['main_idea'])}</p>")
    groups = defaultdict(list)
    for image in result["images"]:
        groups[image["group"]].append(image)
    ordered_groups = sorted(groups.items(), key=lambda item: -max(i["score"] for i in item[1]))
    st.subheader("Как ролик работает на имиджи")
    for score in sorted({i["score"] for i in result["images"]}, reverse=True):
        names = [escape(i["label"]) for i in result["images"] if i["score"] == score]
        st.html(f'<p><b>{LABELS[score]}</b> · {"; ".join(names)}</p>')
    st.subheader("UVP и связанные с ними имиджи")
    for group, images in ordered_groups:
        st.markdown(f"#### {group}")
        for image in sorted(images, key=lambda x: -x["score"]):
            score = image["score"]
            background, color = COLORS.get(score, ("#F9E2DF", "#80382E"))
            review = " · Требует проверки" if image["review"] else ""
            with st.container(border=True):
                st.html(
                    f'<div style="font-size:1.08rem;font-weight:650">{escape(image["label"])}</div>'
                    f'<span style="display:inline-block;margin:.25rem 0 .4rem;background:{background};'
                    f'color:{color};border-radius:.3rem;padding:.12rem .5rem">'
                    f'{LABELS[score]} · {score}</span> · {CLARITY[image["clarity"]]}{review}'
                    f'<p>{escape(image["explanation"])}</p>'
                )
                with st.expander("На чём основана оценка"):
                    for e in image["evidence"]:
                        time = escape(str(e.get("time") or ""))
                        st.html(f'<p><b>{time}</b> {escape(e["observation"])}</p>')
                    if image.get("limitation"):
                        st.html(f'<p>{escape(image["limitation"])}</p>')
                    if image.get("secondary_groups"):
                        st.caption("Также связано с UVP «" + "», «".join(image["secondary_groups"]) + "».")
    with st.expander("Как читать оценки"):
        st.write("Шкала показывает, насколько убедительно ролик связывает имидж с Авито. "
                 "Минус означает, что в ролике есть основание для противоположного впечатления. "
                 "Пометка «Требует проверки» появляется, если возможны разные прочтения "
                 "или остались вопросы к обоснованию.")


def main():
    st.title("UVP")
    st.write("Загрузите ролик и выберите вертикаль. В разборе будет видно, какие обещания Авито "
             "ролик передаёт убедительно и где связь с брендом слабее.")
    vertical = st.selectbox("Вертикаль", VERTICALS, index=None, placeholder="Выберите вертикаль",
                            key="uvp_test_vertical")
    uploaded = st.file_uploader("Ролик (MP4 / MOV)", type=["mp4", "mov"], key="uvp_test_upload")
    run = st.button("Проанализировать ролик", type="primary", disabled=uploaded is None or not vertical,
                    key="uvp_test_run")
    if uploaded is None or not vertical:
        return
    with uploaded.getbuffer() as buf:
        fingerprint = hashlib.sha256(buf).hexdigest()
    identity = (st.session_state.get("mm_user"), vertical, fingerprint)
    if run:
        key = api_key()
        if not key:
            st.error("Анализ временно недоступен. Обратитесь к администратору приложения.")
            return
        notice = st.empty()
        with st.status("Готовим разбор", expanded=True) as status:
            def queued(position):
                if position:
                    notice.warning(f"Сервер занят. Ваш номер в очереди – {position}. "
                                   "Анализ начнётся автоматически.")
                    status.update(label="Ожидаем своей очереди")
                else:
                    notice.empty()

            try:
                result = analyze(uploaded, vertical, key, queued, lambda msg: status.update(label=msg))
            except (AnalysisBusy, AnalysisTimeout):
                status.update(label="Время ожидания истекло", state="error")
                st.warning("Повторите запуск позже.")
                return
            except Exception as err:
                LOGGER.warning("UVP pretest failed (%s)", type(err).__name__)
                status.update(label="Не удалось завершить анализ", state="error")
                st.error("Попробуйте запустить анализ ещё раз. Если ошибка повторится, "
                         "передайте название ролика и вертикаль администратору.")
                return
            finally:
                notice.empty()
            st.session_state["uvp_test_result"] = (identity, result)
            status.update(label="Разбор готов", state="complete", expanded=False)
    saved = st.session_state.get("uvp_test_result")
    if saved and saved[0] == identity:
        show_result(saved[1])


if __name__ in {"__page__", "__main__"}:
    main()
