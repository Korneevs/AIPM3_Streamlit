"""Exercise the manager screen with saved data and incomplete diagnostics."""
from copy import deepcopy

from streamlit.testing.v1 import AppTest


def result():
    return {
        "aipm3": {"index_100": 54, "label": "Ниже среднего", "level": 0},
        "aipm1": {"percentile": 55}, "aipm2": {"percentile": 38},
        "message_delivery": {"percentile": 7},
        "main_idea": "Идея А",
        "diagnostic_recovery": [
            {"respondent_uid": str(i), "condition_group": "full", "answer_type": kind,
             "canonical_idea": idea, "raw_answer": idea}
            for i, (kind, idea) in enumerate([
                ("valid", "Идея А"), ("valid", "Идея Б"),
                ("brand_only", "<BRAND_ONLY>"), ("no_idea", "<NO_IDEA>"),
            ])
        ],
        "feature_profile": {"measurements": [], "withheld": []},
    }


def screen(data):
    app = AppTest.from_string('''
import streamlit as st
from aipm3.result_ui import show_manager_result
result = st.session_state['result']
explanation, profile = show_manager_result(result, saved_profile=result['feature_profile'])
st.session_state['shown_explanation'] = explanation
''')
    app.session_state['result'] = data
    return app.run(timeout=30)


def test_saved_result_renders_all_readings_without_changing_scores():
    data = result()
    before = deepcopy(data)
    app = screen(data)
    assert not app.exception
    assert data == before
    assert app.metric[0].value == "54"
    assert len(app.dataframe[0].value) == 4
    evidence = app.session_state['shown_explanation']['main_message']
    assert len(evidence['top_ideas']) == 2 and evidence['majority'] is False
    assert len(app.get('plotly_chart')) == 1


def test_absent_and_partial_message_data_do_not_turn_into_certain_idea():
    for rows in [[], [result()['diagnostic_recovery'][0]] * 2]:
        data = result()
        data['diagnostic_recovery'] = rows
        app = screen(data)
        assert not app.exception
        assert app.session_state['shown_explanation']['main_message']['status'] != 'available'
        assert app.session_state['shown_explanation']['main_message']['majority'] is None


def test_partial_diagnostics_show_the_reason_even_when_every_row_is_classified():
    data = result()
    data['diagnostic_recovery'][0].pop('respondent_uid')
    app = screen(data)
    assert not app.exception
    evidence = app.session_state['shown_explanation']['main_message']
    assert evidence['full_count'] == evidence['classified_count']
    assert evidence['status'] == 'partial'
    assert all(any(issue in caption.value for caption in app.caption) for issue in evidence['issues'])
