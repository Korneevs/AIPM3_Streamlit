"""Offline integration against independently exported full-model predictions."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aipm3 import latest_runtime as runtime
from aipm3.latest_manual_inputs import with_celebrity_review

FIXTURES = json.loads((Path(__file__).parent / 'fixtures/aipm0719.json').read_text())


@pytest.mark.parametrize('fixture', FIXTURES,
                         ids=[f"{f['material_kind']}-{f['repeat_count']}" for f in FIXTURES])
def test_export_parity_including_repeat_aggregation_and_classes(fixture):
    for expected in fixture['expected_scores']:
        rows = {task: [row for row in values if row['record'] == expected['record']]
                for task, values in fixture['frames'].items()}
        result = runtime.score_feature_rows(rows, material_kind=fixture['material_kind'],
                                            repeat_count=fixture['repeat_count'])
        for name, value in result['scores'].items():
            assert value == pytest.approx(expected[name], rel=1e-12, abs=1e-14)
        assert result['classes'] == expected['classes']
        assert runtime.validate_cached_result(result)['scores'] == result['scores']
        repeat_expected = [row for row in fixture['expected_per_repeat']
                           if row['record'] == expected['record']]
        for actual, saved in zip(result['per_repeat'], repeat_expected):
            for name in result['scores']:
                assert actual[name] == pytest.approx(saved[name], rel=1e-12, abs=1e-14)


def one_result(kind):
    fixture = next(f for f in FIXTURES if f['material_kind'] == kind and f['repeat_count'] == 5)
    sha = fixture['frames']['n'][0]['sha']
    return runtime.score_feature_rows(fixture['frames'], metadata={'source_sha': sha},
                                      material_kind=kind, repeat_count=5)


@pytest.mark.parametrize('kind', ['finished', 'neuromatics'])
def test_manual_celebrity_is_input_only_and_cache_version_is_strict(kind):
    result = one_result(kind)
    yes = with_celebrity_review(result, True)
    no = with_celebrity_review(result, False)
    assert yes['scores']['norm_ad_recall'] == no['scores']['norm_ad_recall']
    assert yes['scores']['message_delivery'] == no['scores']['message_delivery']
    assert runtime.validate_cached_result(yes)['scores'] == yes['scores']
    wrong = deepcopy(result)
    wrong['scoring_version'] = 'three-heads-no-screen-number-20261008'
    with pytest.raises(ValueError):
        runtime.validate_cached_result(wrong)


@pytest.mark.parametrize('task', ['n', 'm', 'r', 'q'])
def test_exact_global_class_boundaries_without_record_pins(task):
    model = runtime.load_models()
    low, high = runtime.display_reference()[task]['cuts']
    values = [np.nextafter(low, -np.inf), low, np.nextafter(low, np.inf),
              high, np.nextafter(high, np.inf)]
    assert list(model.classify(task, values)) == [1, 1, 2, 2, 3]


@pytest.mark.parametrize('kind', ['finished', 'neuromatics'])
def test_interpretation_matches_mixture_and_selected_display_classes(kind):
    from aipm3.latest_interpretation import build_latest_interpretation
    result = one_result(kind)
    out = build_latest_interpretation(result)
    for card in out['cards']:
        task = card['task']
        assert card['score'] == pytest.approx(result['scores'][runtime.SCORE_NAMES[task]], abs=1e-12)
        assert out['details'][task]['additivity_error'] < 1e-10
        assert card['class'] == result['classes'][task]
        assert card['baseline'] == runtime.display_reference()[task]['mean']
    assert out['overall']['class'] == result['classes']['q']


def test_precision_has_exact_fixed_claim_three_readings_and_own_cache(tmp_path, monkeypatch):
    from aipm3 import content_precision as precision, latest_pipeline
    from aipm3.runtime_resources import file_sha256
    source = tmp_path / 'video.mp4'
    source.write_bytes(b'offline video fixture')
    calls = []

    def request(kwargs, path, **options):
        calls.append((kwargs, path))
        assert options['allow_live'] is False
        if len(calls) == 1:
            result = {'main_claim': 'Дословный посыл', 'evidence': '0:01'}
        else:
            result = dict(main_claim='Дословный посыл', claim=True, function=True,
                          details=[], score=2, reason='Функция без уточнений')
        options['validate'](result)
        return result

    monkeypatch.setattr(latest_pipeline, '_request', request)
    actual = precision.measure(source, tmp_path / 'cache', prepared_sha=file_sha256(source),
                               source_sha='original', api_key='', allow_live=False)
    assert actual['mean'] == 2 and len(calls) == 4
    assert calls[0][0]['messages'][0]['content'][0]['text'] == precision.CONTRACT['claim_prompt']
    assert all(c[0]['messages'][0]['content'][0]['text'] == precision.CONTRACT['prompt'] +
               '\nfixed_main_claim=Дословный посыл' for c in calls[1:])
    assert all(c[0]['model'] == precision.CONTRACT['model'] for c in calls)
    assert len({str(c[1]) for c in calls}) == 4
