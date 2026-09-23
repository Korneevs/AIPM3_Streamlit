"""Fixed-scale group scores; never invoke extraction, prediction or an API."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from aipm3 import feature_profile as profile, models
from aipm3.message_delivery_runtime import FINAL_FEATURES


def manifest():
    return {
        'profile_version': profile.VERSION,
        'scoring_version': models.SCORING_VERSION,
        'model_sha256': deepcopy(models.EXPECTED_ARTIFACT_SHA256),
        'calibration_id': 'synthetic-independent-reference',
        'components': {name: {'scale': scale}
                       for name, scale in [('aipm1', 2.), ('aipm2', .2), ('message_delivery', .5)]},
        'validation': {'completed': False},
    }


def source_result():
    result = {'model_sha256': deepcopy(models.EXPECTED_ARTIFACT_SHA256),
              'scoring_version': models.SCORING_VERSION,
              'aipm3': {'index': 1.2345, 'level': 1}}
    for component, (_, groups) in profile.GROUPS.items():
        result[component] = {'feature_effects': {feature: 0. for features in groups.values() for feature in features}}
    result['aipm2']['feature_effects']['brand_mean_adrecall'] = 1000.
    return result


def set_group_effect(result, component, label, effect):
    features = profile.GROUPS[component][1][label]
    for feature in features:
        result[component]['feature_effects'][feature] = 0.
    result[component]['feature_effects'][features[0]] = effect


def keyed(result):
    return {(r['component'], r['label']): r for r in result['groups']}


@pytest.mark.parametrize('scale', [.001, .2, 1., 100.])
def test_signed_score_is_monotonic_symmetric_bounded_and_neutral_at_zero(scale):
    effects = [-1000 * scale, -3 * scale, -.01 * scale, 0, .01 * scale, 3 * scale, 1000 * scale]
    scores = [profile.score_effect(effect, scale) for effect in effects]
    assert scores == sorted(scores)
    assert all(0 <= value <= 100 for value in scores)
    assert scores[3] == 50
    assert all(value < 50 for value in scores[:3])
    assert all(value > 50 for value in scores[4:])
    assert profile.score_effect(.3 * scale, scale) + profile.score_effect(-.3 * scale, scale) == pytest.approx(100)


@pytest.mark.parametrize('effect,scale', [
    (float('nan'), 1), (float('inf'), 1), (-float('inf'), 1),
    (0, 0), (1, -1), (1, float('nan')), (1, float('inf')),
])
def test_invalid_scales_or_effects_do_not_become_scores(effect, scale):
    with pytest.raises(ValueError):
        profile.score_effect(effect, scale)


def test_exact_eleven_original_groups_cover_frozen_features_without_fixed_brand():
    rows = profile.build_profile(source_result(), manifest())['groups']
    assert len(rows) == 11
    labels = {
        'aipm1': {'Фокус внимания', 'Эмоциональный крючок', 'Выделение оффера', 'Фокус сообщения'},
        'aipm2': {'Видимость бренда', 'Аудиальный крючок', 'Подача и темп'},
        'message_delivery': {'Полнота и конкретность предложения', 'Устойчивость при неполном просмотре',
                             'Речь и автономность звука', 'Смысловой фокус и CTA'},
    }
    expected = {'aipm1': set(models.AIPM1_FEATURES),
                'aipm2': set(models.AIPM2_FEATURES) - {'brand_mean_adrecall'},
                'message_delivery': set(FINAL_FEATURES)}
    for component in profile.GROUPS:
        chosen = [r for r in rows if r['component'] == component]
        assert {r['label'] for r in chosen} == labels[component]
        features = [f for row in chosen for f in row['features']]
        assert len(features) == len(set(features))
        assert set(features) == expected[component]
    assert all(r['score'] == 50 and r['effect'] == 0 for r in rows)


def test_group_effect_is_sum_before_fixed_transform():
    result = source_result()
    result['aipm1']['feature_effects'].update(main_character=2., state_transformation=-.5)
    row = keyed(profile.build_profile(result, manifest()))['aipm1', 'Фокус внимания']
    assert row['effect'] == 1.5
    assert row['score'] == profile.score_effect(1.5, 2.)
    assert row['display_score'] == round(row['score'])


def test_no_within_video_renormalization_and_fixed_brand_has_no_group_score():
    first = source_result()
    set_group_effect(first, 'aipm1', 'Фокус внимания', 1.)
    before = keyed(profile.build_profile(first, manifest()))
    second = deepcopy(first)
    set_group_effect(second, 'aipm1', 'Эмоциональный крючок', -1e12)
    second['aipm2']['feature_effects']['brand_mean_adrecall'] = -1e12
    after = keyed(profile.build_profile(second, manifest()))
    assert before['aipm1', 'Фокус внимания'] == after['aipm1', 'Фокус внимания']
    assert all(before[key] == after[key] for key in before if key != ('aipm1', 'Эмоциональный крючок'))


def test_all_groups_in_component_share_one_independent_reference_scale():
    result = source_result()
    calibration = manifest()
    for component, (_, groups) in profile.GROUPS.items():
        for label in groups:
            set_group_effect(result, component, label, .2)
    rows = profile.build_profile(result, calibration)['groups']
    for component in profile.GROUPS:
        scores = [r['score'] for r in rows if r['component'] == component]
        assert len(set(scores)) == 1
        assert scores[0] == profile.score_effect(.2, calibration['components'][component]['scale'])
    assert len({r['score'] for r in rows}) == 3  # Different model-output units.
    calibration['components']['aipm1']['scale'] = 20.
    changed = keyed(profile.build_profile(result, calibration))
    for original in rows:
        row = changed[original['component'], original['label']]
        assert (row['score'] != original['score']) == (original['component'] == 'aipm1')


@pytest.mark.parametrize('bad', ['missing', None, float('nan'), float('inf'), 'not a number'])
def test_missing_or_nonfinite_effect_is_no_data_not_neutral(bad):
    result = source_result()
    if bad == 'missing':
        del result['aipm1']['feature_effects']['humor']
    else:
        result['aipm1']['feature_effects']['humor'] = bad
    rows = keyed(profile.build_profile(result, manifest()))
    withheld = rows['aipm1', 'Эмоциональный крючок']
    assert withheld['score'] is None and withheld['effect'] is None
    assert 'display_score' not in withheld and withheld['reason']
    assert all(r['score'] == 50 for key, r in rows.items() if key != ('aipm1', 'Эмоциональный крючок'))


@pytest.mark.parametrize('field', ['profile_version', 'scoring_version', 'model_sha256'])
def test_wrong_calibration_contract_is_rejected(field):
    calibration = manifest()
    calibration[field] = 'different'
    with pytest.raises(ValueError, match='frozen models'):
        profile.build_profile(source_result(), calibration)


@pytest.mark.parametrize('field', ['scoring_version', 'model_sha256'])
@pytest.mark.parametrize('missing', [False, True])
def test_incompatible_result_has_no_data_instead_of_plausible_scores(field, missing):
    result = source_result()
    if missing:
        result.pop(field)
    else:
        result[field] = 'different'
    rows = profile.build_profile(result, manifest())['groups']
    assert len(rows) == 11
    assert all(r['score'] is None and 'display_score' not in r for r in rows)


def test_calibration_loader_validates_and_freezes_manifest(monkeypatch):
    data = manifest()
    reads = []

    class FakePath:
        def __init__(self, unused):
            pass

        def with_name(self, name):
            assert name == 'group_calibration.json'
            return self

        def read_text(self):
            reads.append(1)
            return json.dumps(data)

    monkeypatch.setattr(profile, 'Path', FakePath)
    profile.calibration_manifest.cache_clear()
    try:
        assert profile.calibration_manifest() == data
        assert profile.calibration_manifest() == data
        assert len(reads) == 1
        data['profile_version'] = 'outdated'
        profile.calibration_manifest.cache_clear()
        with pytest.raises(ValueError, match='frozen models'):
            profile.calibration_manifest()
    finally:
        profile.calibration_manifest.cache_clear()


def test_deterministic_read_only_without_scoring_or_network(monkeypatch):
    import openai

    def forbidden(*args, **kwargs):
        pytest.fail('Profile must not call Gemini or scoring')

    monkeypatch.setattr(openai, 'OpenAI', forbidden)
    for name in ['score_aipm1', 'score_aipm2', 'score_message_delivery', 'aipm3_score']:
        monkeypatch.setattr(models, name, forbidden)
    source, calibration = source_result(), manifest()
    before = deepcopy((source, calibration))
    first = profile.build_profile(source, calibration)
    assert profile.build_profile(source, calibration) == first
    assert (source, calibration) == before
    source.update(diagnostic_panel=[{'anything': 999}], diagnostics={'current_brief': {'uvp': 'x', 'rtb': 'y'}},
                  interpretation={'summary': 'An unrelated narrative'}, aipm3={'index': -999, 'level': 2})
    assert profile.build_profile(source, calibration) == first


def test_real_frozen_calibration_and_saved_baselines_when_available():
    calibration_path = Path(__file__).resolve().parents[1] / 'aipm3/group_calibration.json'
    baseline_dir = Path('/Users/asekorneev/outputs/aipm3_uvp_idea_20260908')
    paths = [baseline_dir / f'video_{i}.json' for i in range(1, 5)]
    if not calibration_path.exists() or not all(path.exists() for path in paths):
        pytest.skip('Frozen calibration/private baseline fixtures not available in this environment')
    calibration = json.loads(calibration_path.read_text())
    assert calibration['profile_version'] == profile.VERSION
    assert calibration['scoring_version'] == models.SCORING_VERSION
    assert calibration['model_sha256'] == models.EXPECTED_ARTIFACT_SHA256
    assert set(calibration['components']) == set(profile.GROUPS)
    for path in paths:
        result = json.loads(path.read_text())
        before = deepcopy(result)
        report = profile.build_profile(result, calibration)
        assert report == profile.build_profile(result, calibration)
        assert len(report['groups']) == 11
        assert all(r['score'] is not None and 0 <= r['score'] <= 100 for r in report['groups'])
        for row in report['groups']:
            expected = sum(result[row['component']]['feature_effects'][f] for f in row['features'])
            assert row['effect'] == pytest.approx(expected)
            assert row['score'] == pytest.approx(profile.score_effect(expected, calibration['components'][row['component']]['scale']))
        assert result == before
