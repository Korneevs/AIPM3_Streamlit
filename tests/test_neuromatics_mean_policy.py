"""Mean replacement must be literal, versioned, and neuromatics-only."""
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aipm3 import latest_runtime as runtime
from aipm3.neuromatics_models import NeuromaticsModels
from aipm3.neuromatics_mean_policy import load_policy


@pytest.fixture
def frames():
    result = {}
    for task in 'nmr':
        history = pd.read_csv(runtime.BUNDLE_DIR / 'data' / f'fit_{task}.csv')
        frame = pd.DataFrame([history.iloc[0].to_dict()] * 3)
        frame['record'] = 'policy-test'
        frame['family'] = 'new-policy-family'
        frame['brand'] = 'Avito'
        frame['vertical'] = 'Goods'
        frame['sha'] = 'c' * 64
        frame['repeat'] = range(1, 4)
        frame['phys__duration'] = frame['total_video_duration_sec'] = 30.
        result[task] = frame
    result['n']['state_transformation'] = 0.
    return result


def test_user_rule_and_equal_video_means():
    policy = load_policy()
    for row in policy['decisions']:
        expected = ('keep' if row['before'] is None or row['before'] <= 4 else
                    'keep_neuromatics_prompt' if 5 <= row['before'] <= 7 and row['after'] <= 4 else
                    'finished_mean')
        assert row['action'] == expected
    assert policy['constants']['m'] == {}
    assert sum(map(len, policy['constants'].values())) == 6
    assert next(d for d in policy['decisions'] if d['feature'] == 'jingle_present')['action'] == 'keep_neuromatics_prompt'
    for task, constants in policy['constants'].items():
        if not constants:
            continue
        data = pd.read_csv(runtime.BUNDLE_DIR / 'data' / f'fit_{task}.csv', float_precision='round_trip')
        keys = data['sha'].where(data.sha.notna(), 'record:' + data.record.astype(str))
        for feature, spec in constants.items():
            values = data[feature]
            if spec['scale'] == 'fraction_of_duration':
                values = values / data.phys__duration.clip(lower=1)
            means = values.groupby(keys).mean().dropna()
            assert spec['video_count'] == len(means)
            assert spec['model_input_mean'] == pytest.approx(means.mean(), abs=1e-14)


def test_constant_inputs_cannot_change_scores_and_are_not_rounded(frames):
    model = runtime.load_models('neuromatics')
    policy = load_policy()
    for task in ['n', 'r']:
        original = frames[task].copy(deep=True)
        changed = original.copy(deep=True)
        for feature in policy['constants'][task]:
            changed[feature] = [0., 1., 100.]
        np.testing.assert_array_equal(model.heads[task].predict(original), model.heads[task].predict(changed))
    index = model.heads['r'].state['columns'].index('scene_pace_high')
    expected = policy['constants']['r']['scene_pace_high']['model_input_mean']
    assert 0 < expected < 1
    assert np.all(model.heads['r'].raw(frames['r'])[:, index] == expected)


def test_noticeability_matches_direct_mean_substitution(frames):
    frozen = runtime.load_latest_models()
    legacy = NeuromaticsModels(frozen, runtime.model_module().coefficient, use_mean_policy=False)
    policy = load_policy()
    adjusted = legacy.heads['n'].transformed(frames['n'])
    for feature, spec in policy['constants']['n'].items():
        value = spec['model_input_mean']
        adjusted[feature] = value * adjusted.phys__duration if spec['scale'] == 'fraction_of_duration' else value
    expected = frozen.heads['n'].predict(adjusted)
    actual = runtime.load_models('neuromatics').heads['n'].predict(frames['n'])
    np.testing.assert_array_equal(actual, expected)


def test_finished_and_message_heads_preserved_and_old_results_rejected(frames):
    before = {t: d.copy(deep=True) for t, d in frames.items()}
    frozen = runtime.load_latest_models()
    expected = {t: frozen.heads[t].predict(d) for t, d in frames.items()}
    current = runtime.score_feature_rows(frames, material_kind='neuromatics', repeat_count=3)
    assert current['scoring_version'] == load_policy()['scoring_version']
    np.testing.assert_array_equal(pd.DataFrame(current['per_repeat']).message_delivery, expected['m'])
    finished = runtime.score_feature_rows(frames, material_kind='finished', repeat_count=3)
    for task, name in runtime.SCORE_NAMES.items():
        np.testing.assert_array_equal(pd.DataFrame(finished['per_repeat'])[name], expected[task])
        pd.testing.assert_frame_equal(frames[task], before[task])
    assert runtime.validate_cached_result(current)['scores'] == current['scores']
    old = deepcopy(current); old['scoring_version'] = 'neuromatics-no-screen-number-20261008'
    with pytest.raises(ValueError):
        runtime.validate_cached_result(old)


def test_fixed_inputs_do_not_appear_as_video_findings(frames):
    from aipm3.latest_interpretation import build_latest_interpretation
    result = runtime.score_feature_rows(frames, material_kind='neuromatics', repeat_count=3)
    explanation = build_latest_interpretation(result)
    for task, constants in load_policy()['constants'].items():
        details = explanation['details'][task]
        assert not set(constants) & {d['feature'] for d in details['drivers']}
        assert details['additivity_error'] < 1e-9
        for feature in constants:
            index = details['feature_order'].index(feature)
            np.testing.assert_allclose(np.asarray(details['per_repeat_shap'])[:, index], 0, atol=1e-12)
