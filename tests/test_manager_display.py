from copy import deepcopy
import pytest
from aipm3.manager_display import with_display_classes, REFERENCE


def source(kind='neuromatics'):
    return dict(material_kind=kind, overall=dict(score=.0107, level='Типичный уровень', index=96),
                cards=[dict(task=t, score=s, level='Типичный уровень', index=100) for t, s in
                       [('n', .36), ('r', .1084), ('m', .25)]],
                details={'unchanged': True}, manager_semantic={'unchanged': True})


def test_recalibration_changes_only_class_colours_for_neuromatics():
    original = source(); before = deepcopy(original)
    updated = with_display_classes(original, REFERENCE['scoring_version'])
    assert updated['overall']['level'] == 'Выше типичного уровня'
    assert updated['cards'][0]['level'] == 'Ниже типичного уровня'
    assert updated['cards'][1]['level'] == 'Типичный уровень'
    assert original == before
    updated.pop('display_class_calibration')
    for after, old in zip([updated['overall']] + updated['cards'], [before['overall']] + before['cards']):
        after['level'] = old['level']
    assert updated == before


def test_finished_classes_are_untouched():
    original = source('finished'); before = deepcopy(original)
    assert with_display_classes(original, 'finished-version') == before


def test_scoring_version_must_match_and_exact_boundaries_stay_yellow():
    with pytest.raises(ValueError):
        with_display_classes(source(), 'wrong-version')
    for cut in REFERENCE['references']['n']['cuts']:
        original = source(); original['cards'][0]['score'] = cut
        assert with_display_classes(original, REFERENCE['scoring_version'])['cards'][0]['level'] == 'Типичный уровень'
