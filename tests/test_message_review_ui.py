from streamlit.testing.v1 import AppTest

from aipm3 import message_alignment


def fixture(relation='equivalent', main=('matched', 'matched'), uvp=None):
    uvp = uvp or main
    answers = [{'respondent_id': str(i), 'answer': 'Смысл ответа'} for i in range(2)]
    coding = {'relation': {'status': relation, 'reason': 'Основание смыслового сравнения'},
              'answers': [{'respondent_id': a['respondent_id'], **{
                  key: {'status': value, 'quote': a['answer'] if value != 'absent' else ''}
                  for key, value in [('main_idea', main[i]), ('uvp', uvp[i])]}} for i, a in enumerate(answers)]}
    return message_alignment.combine([coding, coding], 'Основная идея', 'Задуманный посыл', answers)


def app(data):
    at = AppTest.from_string('''
import streamlit as st
from aipm3.message_review_ui import show_message_comparison
show_message_comparison(st.session_state['data'], 'test')
''')
    at.session_state['data'] = data
    return at.run()


def test_manager_idea_match_and_gap_are_separate():
    at = app(fixture('different'))
    assert not at.exception
    assert any('другую идею' in m.value for m in at.warning)
    assert any('не означает совпадения' in m.value for m in at.markdown)
    assert len(at.dataframe) == 0
    assert len(at.metric) == 2


def test_partial_relation_does_not_claim_uvp_is_narrower():
    at = app(fixture('partial'))
    assert any('совпадают не полностью' in m.value for m in at.warning)


def test_conflict_does_not_display_confident_rates():
    at = app(fixture('equivalent', uvp=('matched', 'absent')))
    assert not at.exception
    assert all(m.value == '—' for m in at.metric)
    assert any('расходятся' in m.value for m in at.warning)
    assert not any('Уверенно' in c.value for c in at.caption)


def test_zero_both_is_not_successful_readability():
    at = app(fixture(main=('absent', 'absent')))
    assert any('нулевая' in m.value.lower() for m in at.markdown)
    assert not any('Потери относительно основной идеи нет' in m.value for m in at.markdown)


def test_one_sided_uncertainty_does_not_claim_uvp_is_less_frequent():
    data = fixture('partial')
    data['relation'] = {'status': 'uncertain', 'coder_statuses': ['equivalent', 'partial'], 'reasons': []}
    data['gap'] = {'direction': 'uncertain', 'pp': -25, 'lower_pp': -25, 'upper_pp': 0}
    at = app(data)
    assert not at.exception
    assert any('спорна только полнота' in m.value for m in at.warning)
    assert any('Потери UVP относительно основной идеи не видно' in m.value for m in at.markdown)
    assert any('на 25.0 п.п. чаще' in m.value for m in at.markdown)


def test_rtb_parts_need_joint_match_not_separate_majorities():
    parts = []
    for p in range(3):
        rows = [{'respondent_id': str(i), 'status': 'absent' if i in [2 * p, 2 * p + 1] else 'matched'} for i in range(10)]
        parts.append({'dimension': 'rtb', 'brief_quote': str(p), 'n': 10, 'answers': rows,
                      'counts': {'matched': 8, 'absent': 2, 'partial': 0, 'contradicted': 0, 'uncertain': 0}})
    at = AppTest.from_string('''
import streamlit as st
from aipm3.message_review_ui import show_promise_support
show_promise_support({'rtb':'Подтверждение'}, st.session_state['data'])
''')
    at.session_state['data'] = {'summary': parts}
    at.run()
    assert not at.exception
    assert any('4 из 10' in c.value for c in at.caption)
    assert not any('сохраняется в большинстве' in m.value for m in at.markdown)


def promise_app(statuses):
    rows = [{
        'respondent_id': str(i), 'status': status,
        'answer': 'Есть автомобили с пробегом.',
        'missing_brief_quote': 'проверенные' if status == 'partial' else '',
    } for i, status in enumerate(statuses)]
    part = {
        'dimension': 'rtb', 'brief_quote': 'проверенные автомобили с пробегом',
        'n': len(rows), 'answers': rows,
        'counts': {status: statuses.count(status)
                   for status in ['matched', 'partial', 'absent', 'contradicted', 'uncertain']},
    }
    at = AppTest.from_string('''
import streamlit as st
from aipm3.message_review_ui import show_promise_support
show_promise_support({'rtb':'проверенные автомобили с пробегом'}, st.session_state['data'])
''')
    at.session_state['data'] = {'summary': [part]}
    return at.run()


def test_rtb_majority_contradiction_precedes_isolated_missing_detail():
    at = promise_app(['contradicted'] * 10 + ['partial', 'matched'])
    assert not at.exception
    text = [m.value for m in at.markdown]
    contradiction = [i for i, value in enumerate(text) if 'противоположный' in value]
    losses = [i for i, value in enumerate(text) if 'Теряется важная деталь' in value]
    assert contradiction, 'A majority contradiction must not be hidden by one partial answer'
    assert not losses or min(contradiction) < min(losses)


def test_rtb_joint_majority_is_not_headlined_as_general_loss():
    at = promise_app(['matched'] * 10 + ['partial', 'absent'])
    assert not at.exception
    text = [m.value for m in at.markdown]
    assert any('сохраняется в большинстве' in value for value in text)
    assert not any(value.startswith('**Теряется важная деталь:') for value in text)
    assert any('10 из 12' in c.value for c in at.caption)
