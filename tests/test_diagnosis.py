import copy

import pytest
from streamlit.testing.v1 import AppTest

from aipm3.creative_diagnosis import build_diagnosis, panel_signal, scene_example
from aipm3 import creative_review as review


def result():
    return {'source_sha': 'test', 'duration_seconds': 20, 'main_idea': 'Выбор автомобилей',
            'diagnostic_panel': [{'call_id': i // 3, 'main_claim_linguistic_directness': 3,
                                  'cta_clarity': i % 2, 'audio_only_message_completeness': 3,
                                  'visual_only_message_completeness': 2} for i in range(30)],
            'recovery_curve': {'full': .5, 'nested_50_m1': .5, 'nested_50_m2': .5},
            'transcripts': [{'transcript': 'Можно найти любой автомобиль.'}] * 3,
            'diagnostics': {'scene_evidence': {'version': review.VERSION, 'source_sha': 'test', 'scenes': [
                {'quote': 'Можно найти любой автомобиль.', 'start_seconds': 7, 'end_seconds': 10,
                 'groups': ['message_focus']}]}}}


def test_deterministic_no_mutation_and_no_shap_dependence():
    source = result()
    before = copy.deepcopy(source)
    first = build_diagnosis(source)
    assert source == before
    assert first == build_diagnosis(source)
    source['interpretation'] = {'group_rows': [{'Локальное влияние, %': 999}]}
    assert first == build_diagnosis(source)
    source['diagnostics']['current_brief'] = {'uvp': 'other', 'rtb': 'other'}
    assert first == build_diagnosis(source)
    assert any(f['id'] == 'direct' and f['scene'] for f in first['strengths'])
    assert any(f['id'] == 'sound_dependency' for f in first['risks'])


def test_no_invented_consensus_or_flat_low_recovery_strength():
    report = build_diagnosis(result())
    assert report['uncertain']
    assert not any(f['id'] == 'cta' for f in report['risks'])
    assert not any('recovery' in f['id'] for f in report['strengths'])
    assert build_diagnosis({})['strengths'] == []
    assert build_diagnosis({})['risks'] == []
    source = result()
    for row in source['diagnostic_panel']:
        row['call_id'] = 1
    assert not panel_signal(source, 'cta_clarity', lambda v: True)['stable']


def test_no_scene_review_still_uses_verified_transcript_context():
    source = result()
    source['diagnostics'] = {}
    source['main_idea'] = 'Найти автомобиль'
    report = build_diagnosis(source)
    assert next(f for f in report['strengths'] if f['id'] == 'direct')['transcript_quote'] == 'Можно найти любой автомобиль.'


@pytest.mark.parametrize('fault', ['source', 'version', 'quote', 'time', 'transcript'])
def test_scene_requires_source_time_and_two_transcripts(fault):
    source = result()
    evidence = source['diagnostics']['scene_evidence']
    if fault == 'source':
        evidence['source_sha'] = 'wrong'
    elif fault == 'version':
        evidence['version'] = 'old'
    elif fault == 'quote':
        evidence['scenes'][0]['quote'] = 'Доставка за час'
    elif fault == 'time':
        evidence['scenes'][0]['end_seconds'] = 21
    else:
        source['transcripts'] = source['transcripts'][:1]
    assert scene_example(source, ['message_focus']) is None


def test_actual_four_saved_scores_and_features_unchanged():
    import json
    from pathlib import Path
    root = Path.home() / 'outputs'
    if not (root / 'aipm3_review_redesign_20260908/video_4.json').exists():
        pytest.skip('Private saved video results available locally only')
    for i in range(1, 5):
        old = json.loads((root / f'aipm3_four_videos_20260908/video_{i}.json').read_text())
        new = json.loads((root / f'aipm3_review_redesign_20260908/video_{i}.json').read_text())
        for key in ['aipm1', 'aipm2', 'message_delivery', 'aipm3', 'objective_features', 'model_sha256']:
            assert new[key] == old[key]
        before = copy.deepcopy(new)
        report = build_diagnosis(new)
        assert new == before
        assert report == build_diagnosis(new)


def test_manager_ui_has_evidence_no_shap():
    at = AppTest.from_string('''
import streamlit as st
from aipm3.review_ui import show_manager_readout
show_manager_readout(st.session_state['result'])
''')
    at.session_state['result'] = result()
    at.run()
    assert not at.exception
    text = ' '.join(i.value for i in at.markdown) + ' '.join(i.value for i in at.text)
    assert 'Предложение сформулировано прямо' in text
    assert 'Можно найти любой автомобиль.' in text
    assert 'SHAP' not in text and 'Локальное влияние' not in text
    assert len(at.dataframe) == 0
