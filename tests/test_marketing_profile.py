"""Every contribution remains visible, in one outcome and one direction only."""
from copy import deepcopy
import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest
from aipm3 import models
from aipm3.feature_profile import GROUPS
from aipm3.marketing_profile import build_marketing_profile, OBSERVATION_LABELS
from aipm3.profile_ui import topic_figure


def source():
    result = {'scoring_version': models.SCORING_VERSION,
              'model_sha256': deepcopy(models.EXPECTED_ARTIFACT_SHA256)}
    for component, (_, groups) in GROUPS.items():
        result[component] = {'feature_effects': {f: (-1)**i * (i+1)
            for i, f in enumerate(f for names in groups.values() for f in names)}, 'feature_values': {}}
    result['aipm2']['feature_effects']['brand_mean_adrecall'] = 1000
    return result


def rows(profile, component):
    return [t['outcomes'][0] for t in profile['topics'] if t['component'] == component]


def test_every_feature_is_retained_once_in_one_outcome_and_one_sign():
    result = source()
    before = deepcopy(result)
    p = build_marketing_profile(result)
    assert p == build_marketing_profile(result) and result == before
    for component, (_, groups) in GROUPS.items():
        actual = rows(p, component)
        expected = [f for fs in groups.values() for f in fs]
        covered = [f for row in actual for f in row['features']]
        assert len(covered) == len(set(covered)) == len(expected)
        assert set(covered) == set(expected)
        assert sum(row['importance'] for row in actual) == pytest.approx(100)
        assert sum(row['effect'] for row in actual) == pytest.approx(sum(result[component]['feature_effects'][f] for f in expected))
        for row in actual:
            assert not row['mixed']
            assert len({d['direction'] for d in row['drivers']}) == 1
            assert not (row['positive_share'] and row['negative_share'])
            assert sum(d['importance'] for d in row['drivers']) == pytest.approx(row['importance'])
    assert all(len(t['outcomes']) == 1 for t in p['topics'])
    assert set(OBSERVATION_LABELS) == {f for _, groups in GROUPS.values() for fs in groups.values() for f in fs}


def test_opposite_outcomes_and_opposite_features_do_not_cancel_in_one_card():
    r = source()
    r['aipm1']['feature_effects'].update(main_character=4, state_transformation=-4, promo=2)
    p = build_marketing_profile(r)
    story = [t for t in p['topics'] if t['topic_id'] == 'attention']
    assert len(story) == 2 and {t['direction'] for t in story} == {'up', 'down'}
    assert all(t['importance'] > 0 for t in story)
    offer = [t for t in p['topics'] if t['topic_id'] == 'offer']
    complexity = [t for t in p['topics'] if t['topic_id'] == 'complexity']
    assert all(t['component'] == 'aipm1' for t in offer)
    assert all(t['component'] == 'message_delivery' for t in complexity)


def test_cards_rank_negative_contributions_then_positive_within_each_outcome():
    p = build_marketing_profile(source())
    for component in GROUPS:
        rr = rows(p, component)
        directions = [r['direction'] for r in rr]
        assert directions == sorted(directions, key={'down': 0, 'up': 1, 'balanced': 2}.get)
        for direction in {'up', 'down'}:
            weights = [r['importance'] for r in rr if r['direction'] == direction]
            assert weights == sorted(weights, reverse=True)


def test_other_outcomes_context_and_change_of_units_do_not_change_weights():
    r = source()
    before = rows(build_marketing_profile(r), 'aipm2')
    r['aipm2']['feature_effects']['brand_mean_adrecall'] = -1e20
    r['aipm1']['feature_effects']['main_character'] = 1e20
    assert rows(build_marketing_profile(r), 'aipm2') == before
    for f in r['aipm2']['feature_effects']:
        r['aipm2']['feature_effects'][f] *= 1e-18
    after = rows(build_marketing_profile(r), 'aipm2')
    assert [x['direction'] for x in before] == [x['direction'] for x in after]
    assert [x['importance'] for x in before] == pytest.approx([x['importance'] for x in after])


@pytest.mark.parametrize('bad', [None, 'missing', float('nan'), float('inf'), 'invalid'])
def test_missing_effect_withholds_all_weights_of_that_outcome(bad):
    r = source()
    if bad == 'missing':
        del r['aipm1']['feature_effects']['humor']
    else:
        r['aipm1']['feature_effects']['humor'] = bad
    p = build_marketing_profile(r)
    assert all(not row['available'] and 'importance' not in row for row in rows(p, 'aipm1'))
    assert all(row['available'] for row in rows(p, 'aipm2'))


def test_missing_observations_never_become_absence_or_fabricated_conclusion():
    r = source()
    r['aipm1']['feature_values'].update(promo=0, has_screen_offer_text=None)
    r['message_delivery_business'] = {'offer_condition_count': .1, 'cta_clarity': False}
    p = build_marketing_profile(r)
    observed = {label: value for t in p['topics'] for label, _, value in t['observations']}
    assert observed['Акция или бонус'] == 'Не обнаружено'
    assert observed['Конкретное предложение на экране'] == 'Нет данных'
    assert observed['Условия предложения'] == '0,1'
    assert all('Главный герой: не обнаружено' not in t['finding'] for t in p['topics'])


@pytest.mark.parametrize('field', ['scoring_version', 'model_sha256'])
def test_incompatible_results_have_no_weights_or_observed_claims(field):
    r = source(); r[field] = 'wrong'
    p = build_marketing_profile(r)
    assert all(not t['observations'] and not t['outcomes'][0]['available'] for t in p['topics'])


def test_zero_effects_are_neutral_without_fake_ranking():
    r = source()
    for c in GROUPS:
        r[c]['feature_effects'] = dict.fromkeys(r[c]['feature_effects'], 0)
    p = build_marketing_profile(r)
    assert len(p['topics']) == 11
    assert all(t['direction'] == 'balanced' and t['importance'] == 0 for t in p['topics'])
    assert all('не меняют' in t['conclusion'] for t in p['topics'])


def test_plot_has_one_outcome_and_one_direction_per_card():
    for t in build_marketing_profile(source())['topics']:
        fig = topic_figure(t['outcomes'])
        assert list(fig.layout.xaxis.range) == [-100, 100]
        assert len(fig.data[0].x) == len(fig.data[1].x) == 1
        assert not (fig.data[0].x[0] and fig.data[1].x[0])


def test_ui_keeps_all_contributions_in_three_tabs_without_intro():
    r = source()
    r['aipm1']['feature_values']['promo'] = 0
    app = AppTest.from_string('''
import streamlit as st
from aipm3.profile_ui import show_feature_profile
show_feature_profile(st.session_state['result'])
''')
    app.session_state['result'] = r
    app.run(timeout=30)
    assert not app.exception
    assert [t.label for t in app.tabs] == ['Заметность', 'Запоминаемость', 'Считываемость']
    text = ' '.join(x.value for x in app.markdown)
    assert 'С чего начать' not in text
    assert 'Вариант для проверки' in text
    assert len(app.get('plotly_chart')) == len(build_marketing_profile(r)['topics'])
    assert app.session_state['result'] == r


def test_suggestions_do_not_treat_missing_humor_as_an_existing_joke():
    r = source()
    r['aipm1']['feature_effects']['humor'] = -1
    r['aipm1']['feature_values']['humor'] = 0
    topic = next(t for t in build_marketing_profile(r)['topics'] if t['topic_id'] == 'humor')
    assert 'юмористическим моментом' in topic['outcomes'][0]['drivers'][0]['check']
    r['aipm1']['feature_values']['humor'] = 1
    topic = next(t for t in build_marketing_profile(r)['topics'] if t['topic_id'] == 'humor')
    assert 'Сравнить шутку' in topic['outcomes'][0]['drivers'][0]['check']


def test_replay_saved_results_never_runs_inference(monkeypatch):
    import openai
    def forbidden(*a, **kw):
        pytest.fail('Presentation must not call an API or rescore')
    monkeypatch.setattr(openai, 'OpenAI', forbidden)
    for name in ['score_aipm1', 'score_aipm2', 'score_message_delivery', 'aipm3_score']:
        monkeypatch.setattr(models, name, forbidden)
    paths = sorted((Path.home() / 'Downloads').glob('aipm3_result*.json'))
    if not paths:
        pytest.skip('Private saved results unavailable')
    for path in paths:
        r = json.loads(path.read_text()); original = deepcopy(r)
        p = build_marketing_profile(r)
        assert r == original
        for c in GROUPS:
            assert sum(row['importance'] for row in rows(p, c)) == pytest.approx(100)
