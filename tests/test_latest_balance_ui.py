"""The compact chart must preserve model arithmetic and reviewed evidence."""
from copy import deepcopy

import pytest
from streamlit.testing.v1 import AppTest

from aipm3.latest_balance_ui import balance_items, balance_figure, driver_label, explanation_is_partial
from aipm3.interpretation_checker import apply_review, VERSION
from aipm3.neuromatics_manager import build_manager_draft


def driver(feature, value=1, contribution=.01, verified=True, usable=True):
    return dict(feature=feature, label=feature, value=value, contribution=contribution,
                stable_fraction=1, usable=usable, factual_observation=None,
                evidence=dict(verified=verified, observation="Это прямо показано в ролике.", episodes=[]))


def explanation(drivers, adjustment=0):
    expected = .2
    actual = expected + sum(d['contribution'] for d in drivers) + adjustment
    return dict(material_kind="finished", audio_status="complete",
                cards=[dict(task=t, score=actual, baseline=.25, **{'class':2}, explanation_status='partial') for t in 'nmr'],
                details={t: dict(drivers=deepcopy(drivers), expected=expected,
                                 actual=actual, manual_adjustment=adjustment) for t in 'nmr'})


def test_top_four_are_individual_and_remaining_effects_still_reconstruct_raw_score():
    source = explanation([driver('unsupported_large', contribution=-.15, verified=False),
                          *[driver('feature_' + str(i), contribution=.001 * i) for i in range(1,7)]], .07)
    before = deepcopy(source)
    items = balance_items(source, 'r')
    assert [i['driver']['feature'] for i in items if 'driver' in i] == [f'feature_{i}' for i in (6,5,4,3)]
    other = next(i['value'] for i in items if i['label'] == 'Остальные факторы')
    assert other == pytest.approx(-.15 + .001 + .002)
    assert sum(i['value'] for i in items if i['measure'] != 'total') == pytest.approx(source['details']['r']['actual'])
    assert next(i['value'] for i in items if i['label'] == 'Оценка модели') == pytest.approx(.071)
    assert len([i for i in items if i.get('expert')]) == 1
    chart = balance_figure(source, 'r').data[0]
    assert list(chart.measure) == [i['measure'] for i in items]
    assert all(x == 0 for x,m in zip(chart.x, chart.measure) if m == 'total')
    assert source == before


def test_no_qualified_observations_does_not_fill_four_slots():
    source = explanation([driver('weak', usable=False), driver('unverified', verified=False)])
    items = balance_items(source, 'n')
    assert [i['label'] for i in items] == ['База', 'Остальные факторы', 'Итог']
    assert sum(i['value'] for i in items if i['measure'] != 'total') == pytest.approx(.22)


def test_offer_explanation_label_requires_confirmed_low_explanation_need():
    low = driver('panel__offer_novelty_explanation_need', value=0)
    assert driver_label(low) == 'Предложение почти не требует пояснений'
    assert driver_label(dict(low, evidence={'verified':False})) == low['label']
    assert driver_label(dict(low, value=1)) == low['label']


def reviewed_source(action='keep', omit_support=None):
    source = explanation([driver('panel__message_specificity_level', value=3),
                          driver('panel__mandatory_inference_chain_length', value=0)])
    source['manager_semantic'] = build_manager_draft(source)
    draft = source['manager_semantic']
    # Keep this fixture's coverage below the separate rejection floor so that
    # a valid withholding decision can exercise the downstream display gate.
    for coverage in draft['coverage'].values():
        coverage['minimum'] = 1
    items = []
    for task, rows in draft['profiles'].items():
        for row in rows:
            support = list(row['evidence_facts'])
            if task == 'm' and omit_support:
                support.remove(omit_support)
            items.append(dict(id=row['id'], action=action if task == 'm' else 'keep',
                              finding=row['finding'], check=row['check'],
                              direction='balanced', support_ids=support, reason='Проверено.'))
    if action == 'withhold':
        draft['coverage']['m']['supported'] = 0
    receipt = dict(version=VERSION, status='reviewed', fingerprint=draft['fingerprint'], reply={'items':items})
    result = apply_review(source, receipt)
    assert result['manager_semantic']['reviewed'] is True
    return result


def test_valid_withhold_cannot_resurface_in_numeric_profile_explanation():
    source = reviewed_source(action='withhold')
    items = balance_items(source, 'm')
    assert not any('driver' in i for i in items)
    assert next(i['value'] for i in items if i['label']=='Остальные факторы') == pytest.approx(.02)
    assert source['details']['m']['drivers'][0]['usable'] is True
    assert explanation_is_partial(source, 'm', {'explanation_status':'supported'})


def test_top_four_limit_is_not_a_partial_evidence_warning():
    source = explanation([driver('supported_' + str(i)) for i in range(6)])
    assert not explanation_is_partial(source, 'm', {'explanation_status':'supported'})


def test_numeric_audit_uses_neutral_label_for_unadmitted_low_explanation_need():
    source = explanation([driver('panel__offer_novelty_explanation_need', value=0)])
    source['manager_semantic'] = dict(reviewed=True, profiles={task:[] for task in 'nmr'})
    app = AppTest.from_string('''
import streamlit as st
from aipm3.latest_balance_ui import show_balance_profile
show_balance_profile(st.session_state['explanation'])
''')
    app.session_state['explanation'] = source
    app.run()
    assert not app.exception
    for table in app.dataframe:
        assert list(table.value['Признак']) == ['Сложность объяснения предложения']
        assert not table.value['Допущен к объяснению'].any()
    assert 'Предложение почти не требует пояснений' not in ' '.join(x.value for x in app.markdown)


def test_combined_editorial_row_only_allows_individual_cited_support():
    omitted = 'panel__mandatory_inference_chain_length'
    source = reviewed_source(omit_support=omitted)
    items = balance_items(source, 'm')
    assert [i['driver']['feature'] for i in items if 'driver' in i] == ['panel__message_specificity_level']
    assert next(i['value'] for i in items if i['label']=='Остальные факторы') == pytest.approx(.01)


@pytest.mark.parametrize('cause', ['missing_row', 'different_input', 'opposite_sign'])
def test_reviewed_profile_fails_closed_on_missing_or_inconsistent_feature_support(cause):
    source = reviewed_source()
    row = source['manager_semantic']['profiles']['m'][0]
    if cause == 'missing_row':
        source['manager_semantic']['profiles']['m'] = []
    elif cause == 'different_input':
        for fact in row['evidence_facts'].values():
            fact['same_input'] = False
    else:
        row['direction'] = 'down'
    items = balance_items(source, 'm')
    assert not any('driver' in item for item in items)
    assert next(i['value'] for i in items if i['label']=='Остальные факторы') == pytest.approx(.02)


@pytest.mark.parametrize('available', [True, False])
def test_streamlit_chart_keeps_evidence_collapsed_and_suppresses_unavailable_scores(available):
    source = explanation([driver('panel__offer_novelty_explanation_need', value=0)])
    for card in source['cards']:
        card['assessment_available'] = available
    app = AppTest.from_string('''
import streamlit as st
from aipm3.latest_balance_ui import show_balance_profile
show_balance_profile(st.session_state['explanation'])
''')
    app.session_state['explanation'] = source
    app.run()
    assert not app.exception
    assert len(app.get('plotly_chart')) == (3 if available else 0)
    if available:
        assert len(app.expander) == 6
        assert all(not expander.proto.expanded for expander in app.expander)
        assert 'Баланс оценки' in [x.value for x in app.subheader]
        assert sum('Разбор частичный' in x.value for x in app.caption) == 3
        assert sum('База графика: 0,2000. Ориентир 0% в верхней карточке: 0,2500.' in x.value
                   for x in app.caption) == 3
    else:
        assert not app.expander
        assert sum('Для оценки нужна полная озвучка.' == x.value for x in app.caption) == 3
