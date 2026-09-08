"""Native-unit profile contracts; none of these tests request model inference."""
import copy
import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from aipm3 import feature_profile as profile
from aipm3.profile_ui import group_figure


def source_result():
    return {
        'duration_seconds': 20,
        'objective_runs': {
            'aipm1': [dict(main_character=True, state_transformation=False, humor=True)
                      for _ in range(3)],
            'aipm2': [dict(brand_logo_screen_seconds=10, pack_shot_duration_seconds=4)
                      for _ in range(2)],
        },
        'diagnostic_panel': [
            dict(call_id=(i - 1) // 3 + 1, respondent_id=f'p{i:02d}',
                 core_claim_quartile_coverage=3, offer_condition_count=1,
                 audio_only_message_completeness=3, visual_only_message_completeness=2,
                 cta_clarity=False, message_specificity_level=2)
            for i in range(1, 31)
        ],
    }


def measured(result):
    return {row['feature']: row for row in profile.measure_profile(result)}


def test_native_units_and_separate_constructs_are_preserved():
    rows = measured(source_result())
    assert rows['brand_logo_screen_seconds']['value'] == 50
    assert rows['brand_logo_screen_seconds']['native_value'] == 10
    assert rows['pack_shot_duration_seconds']['value'] == 20
    assert rows['core_claim_quartile_coverage']['value'] == 3
    assert rows['core_claim_quartile_coverage']['maximum'] == 4
    assert rows['audio_only_message_completeness']['value'] == 3
    assert rows['audio_only_message_completeness']['maximum'] == 3
    assert rows['main_character']['display'] == 'Есть'
    assert rows['state_transformation']['display'] == 'Нет'
    assert all(rows[f]['calls'] == 10 for f in profile.CANDIDATES
               if profile.CANDIDATES[f][2] == 'panel')
    result = profile.build_profile(source_result())
    assert 'score' not in result and 'group_scores' not in result
    assert all('weight' not in r and 'effect' not in r for r in result['measurements'])


def test_deterministic_read_only_without_scoring_or_network(monkeypatch):
    import openai
    from aipm3 import models

    def forbidden(*args, **kwargs):
        pytest.fail('Profile must not call Gemini or scoring')

    monkeypatch.setattr(openai, 'OpenAI', forbidden)
    for name in ['score_aipm1', 'score_aipm2', 'score_message_delivery', 'aipm3_score']:
        monkeypatch.setattr(models, name, forbidden)
    source = source_result()
    before = copy.deepcopy(source)
    expected = profile.build_profile(source)
    assert source == before
    assert profile.build_profile(source) == expected
    source.update(aipm1={'feature_effects': {'humor': 999}},
                  aipm2={'score': -99}, message_delivery={'feature_effects': {'x': 999}},
                  aipm3={'index': 888}, interpretation={'summary': 'Invented advice'},
                  diagnostics={'current_brief': {'uvp': 'anything', 'rtb': 'anything'}})
    assert profile.build_profile(source) == expected


@pytest.mark.parametrize('fault', ['missing', 'extra', 'duplicate', 'wrong_call',
                                  'string_call', 'float_call', 'boolean_call', 'non_row'])
def test_incomplete_or_malformed_panel_is_not_a_small_valid_panel(fault):
    source = source_result()
    rows = source['diagnostic_panel']
    if fault == 'missing':
        rows.pop()
    elif fault == 'extra':
        rows.append(copy.deepcopy(rows[0]))
    elif fault == 'duplicate':
        rows[1]['respondent_id'] = rows[0]['respondent_id']
    elif fault == 'wrong_call':
        rows[0]['call_id'] = 2
    elif fault == 'string_call':
        rows[0]['call_id'] = '1'
    elif fault == 'float_call':
        rows[0]['call_id'] = 1.0
    elif fault == 'boolean_call':
        rows[0]['call_id'] = True
    else:
        rows[0] = None
    report = profile.build_profile(source)
    assert not any(row['source'] == 'panel' for row in report['measurements'])
    assert any(row['source'] == 'panel' for row in report['withheld'])
    assert any(row['source'] == 'aipm1' for row in report['measurements'])


@pytest.mark.parametrize('feature,values', [
    ('core_claim_quartile_coverage', [5, 1, 1]),
    ('core_claim_quartile_coverage', [-1, 4, 4]),
    ('audio_only_message_completeness', [4, 1, 1]),
    ('audio_only_message_completeness', [-1, 3, 3]),
    ('offer_condition_count', [1.5, 1, 1]),
    ('visual_only_message_completeness', [float('nan'), 2, 2]),
    ('cta_clarity', [2, 0, 0]),
    ('core_claim_quartile_coverage', [None, 2, 2]),
])
def test_bad_raw_values_cannot_hide_inside_valid_call_average(feature, values):
    source = source_result()
    for row, value in zip(source['diagnostic_panel'], values):
        row[feature] = value
    item = measured(source)[feature]
    assert item['value'] is None
    assert not item['within_stable']


def test_one_feature_missing_does_not_invalidate_unrelated_measurements():
    source = source_result()
    del source['diagnostic_panel'][0]['offer_condition_count']
    rows = measured(source)
    assert rows['offer_condition_count']['value'] is None
    assert rows['audio_only_message_completeness']['value'] == 3


@pytest.mark.parametrize('component,count,feature', [
    ('aipm1', 2, 'main_character'), ('aipm1', 4, 'main_character'),
    ('aipm2', 1, 'brand_logo_screen_seconds'), ('aipm2', 3, 'brand_logo_screen_seconds'),
])
def test_legacy_extraction_requires_exact_repeat_count(component, count, feature):
    source = source_result()
    row = source['objective_runs'][component][0]
    source['objective_runs'][component] = [copy.deepcopy(row) for _ in range(count)]
    assert measured(source)[feature]['value'] is None
    assert measured(source)['visual_only_message_completeness']['value'] == 2


def test_binary_disagreement_is_withheld_instead_of_majority_quality():
    source = source_result()
    source['objective_runs']['aipm1'][0]['humor'] = False
    item = measured(source)['humor']
    assert not item['within_stable']
    assert item['display'] == 'Неоднозначно'
    report = profile.build_profile(source)
    assert 'humor' not in {r['feature'] for r in report['measurements']}
    assert 'humor' in {r['feature'] for r in report['withheld']}


@pytest.mark.parametrize('bad', [-1, 2, .5, float('inf'), None])
def test_binary_raw_values_obey_binary_domain(bad):
    source = source_result()
    source['objective_runs']['aipm1'][0]['main_character'] = bad
    assert measured(source)['main_character']['value'] is None


@pytest.mark.parametrize('seconds,expected', [(18, 100 * 18 / 18.84), (19, 100), (20, None)])
def test_duration_rounding_excess_only_is_allowed(seconds, expected):
    source = source_result()
    source['duration_seconds'] = 18.84
    for row in source['objective_runs']['aipm2']:
        row['brand_logo_screen_seconds'] = seconds
    item = measured(source)['brand_logo_screen_seconds']
    if expected is None:
        assert item['value'] is None
    else:
        assert item['value'] == pytest.approx(expected)
        assert item['native_value'] == seconds


@pytest.mark.parametrize('duration', [0, -1, None, float('nan'), float('inf')])
def test_invalid_duration_does_not_produce_a_brand_percentage(duration):
    source = source_result()
    source['duration_seconds'] = duration
    assert measured(source)['brand_logo_screen_seconds']['value'] is None


def test_brand_call_disagreement_is_shown_as_observed_range_not_a_stability_verdict():
    source = source_result()
    source['objective_runs']['aipm2'][0]['brand_logo_screen_seconds'] = 0
    source['objective_runs']['aipm2'][1]['brand_logo_screen_seconds'] = 20
    item = measured(source)['brand_logo_screen_seconds']
    assert item['low'] == 0 and item['high'] == 100
    assert item['within_stable'] is None
    report = profile.build_profile(source)
    assert 'brand_logo_screen_seconds' in {r['feature'] for r in report['measurements']}
    assert 'pack_shot_duration_seconds' in {r['feature'] for r in report['measurements']}


@pytest.mark.parametrize('changed_calls,changed_value', [
    (1, 2), (1, 0), (5, 0), (0, 0),
])
def test_panel_persona_variability_remains_visible_without_fake_repeatability_claim(
        changed_calls, changed_value):
    source = source_result()
    for row in source['diagnostic_panel'][:3 * changed_calls]:
        row['audio_only_message_completeness'] = changed_value
    item = measured(source)['audio_only_message_completeness']
    assert item['low'] == (changed_value if changed_calls else 3)
    assert item['high'] == 3
    # Persona variation must not be conflated with independent-run validation.
    assert item['within_stable'] is None
    report = profile.build_profile(source)
    assert 'audio_only_message_completeness' in {r['feature'] for r in report['measurements']}
    assert not any(key in item for key in ['confidence_interval', 'human_readability', 'p_value',
                                          'mean_sensitivity_2se', 'max_single_call_influence'])


def test_unvalidated_or_wrong_version_manifest_cannot_approve_a_measurement(monkeypatch):
    manifest = copy.deepcopy(profile.validation_manifest())
    manifest['approved_features'].remove('core_claim_quartile_coverage')
    monkeypatch.setattr(profile, 'validation_manifest', lambda: manifest)
    result = profile.build_profile(source_result())
    assert 'core_claim_quartile_coverage' in {r['feature'] for r in result['withheld']}
    manifest['profile_version'] = 'other-version'
    with pytest.raises(ValueError, match='version'):
        profile.build_profile(source_result())


@pytest.mark.parametrize('group,maximum', [('structure', 4), ('channels', 3), ('brand', 100)])
def test_plotly_chart_retains_native_axis_and_is_not_quality_percentage(group, maximum):
    rows = [r for r in profile.build_profile(source_result())['measurements'] if r['group'] == group]
    figure = group_figure(rows, maximum)
    assert list(figure.layout.xaxis.tickvals) == [0, maximum]
    assert list(figure.data[1].x) == [r['value'] for r in rows]
    assert list(figure.data[1].error_x.array) == [r['high'] - r['value'] for r in rows]
    if maximum != 100:
        assert all('%' not in str(label) for label in figure.layout.xaxis.ticktext)
        assert all('%' not in annotation.text for annotation in figure.layout.annotations)


def test_compact_ui_has_no_brief_inputs_or_duplicate_plus_minus_narrative():
    app = AppTest.from_string('''
import streamlit as st
from aipm3.profile_ui import show_feature_profile
show_feature_profile(st.session_state['result'])
''')
    app.session_state['result'] = source_result()
    app.run()
    assert not app.exception
    text = ' '.join(str(item.value) for kind in ['markdown', 'caption', 'subheader', 'text']
                    for item in getattr(app, kind))
    for prohibited in ['UVP', 'RTB', 'УВП', 'РТБ', 'Что получилось', 'Что ослабляет',
                       'Локальное влияние', 'SHAP']:
        assert prohibited not in text
    assert not app.text_input and not app.text_area and not app.button and not app.dataframe
    assert len(app.get('plotly_chart')) == 3
    assert 'Больше — не всегда лучше' in text


def test_public_manifest_matches_pilot_scope_and_withholds_failed_cta():
    manifest = profile.validation_manifest()
    assert manifest['videos'] == 3
    assert manifest['fresh_repeats_per_video'] == 2
    assert manifest['comparisons_include_saved_baseline'] is True
    assert manifest['tolerance_fraction'] == .1
    assert 'Not recovery or whole-model output' in manifest['scope']
    assert 'cta_clarity' not in manifest['approved_features']
    for feature in manifest['approved_features']:
        assert manifest['measurements'][feature]['passed']
        assert manifest['measurements'][feature]['max_normalized_spread'] <= .1 + 1e-9


def test_saved_three_video_repeats_validate_visible_profile_not_frozen_core_outputs():
    root = Path.home() / 'outputs'
    repeat_dir = root / 'aipm3_compact_profile_20260908'
    baseline_dir = root / 'aipm3_uvp_idea_20260908'
    paths = [repeat_dir / f'video_{i}_repeat_{j}.json' for i in range(1, 4) for j in (1, 2)]
    paths += [baseline_dir / f'video_{i}.json' for i in range(1, 4)]
    if not all(path.exists() for path in paths):
        pytest.skip('Private fresh extraction pilot available locally only')
    visible_features = {f for f, spec in profile.CANDIDATES.items() if spec[0] in profile.GROUPS}
    for video in range(1, 4):
        baseline = json.loads((baseline_dir / f'video_{video}.json').read_text())
        repeats = [json.loads((repeat_dir / f'video_{video}_repeat_{j}.json').read_text()) for j in (1, 2)]
        for run in repeats:
            assert run['fresh'] is True
            assert run['source_sha'] == baseline['source_sha']
            assert run['component_video_sha'] == baseline['component_video_sha']
        measurements = [measured(run) for run in [baseline] + repeats]
        displays = [profile.build_profile(run) for run in [baseline] + repeats]
        assert all({r['feature'] for r in run['measurements']} == visible_features for run in displays)
        assert all(not run['withheld'] for run in displays)
        for feature in visible_features:
            rows = [run[feature] for run in measurements]
            assert all(row['value'] is not None for row in rows)
            values = [row['value'] for row in rows]
            assert (max(values) - min(values)) / rows[0]['maximum'] <= .1 + 1e-9
            if rows[0]['kind'] == 'binary':
                assert len(set(values)) == 1
                assert all(row['within_stable'] for row in rows)
        # Scores were not freshly extracted here: test deliberately makes no claim
        # of full-model repeatability or equality from copied/absent score fields.
