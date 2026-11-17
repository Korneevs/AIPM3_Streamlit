"""Presentation-only classes, calibrated for the fixed-mean prototype scorer."""
from copy import deepcopy
import json
from pathlib import Path

REFERENCE = json.loads((Path(__file__).with_name('manager_display_classes.json')).read_text())
VERSION = REFERENCE['version']


def with_display_classes(interpretation, scoring_version):
    from .latest_runtime import SCORING_VERSION, NEUROMATICS_SCORING_VERSION, release_module
    if scoring_version in (SCORING_VERSION, NEUROMATICS_SCORING_VERSION):
        out = deepcopy(interpretation)
        scores = {c['task']: c['score'] for c in out['cards']}
        scores['q'] = out['overall']['score']
        classes = release_module().score_classes(scores)
        levels = {1: 'Ниже типичного уровня', 2: 'Типичный уровень', 3: 'Выше типичного уровня'}
        for task, card in [('q', out['overall'])] + [(c['task'], c) for c in out['cards']]:
            card['class'] = classes[task]
            card['level'] = levels[classes[task]]
        out['display_class_calibration'] = scoring_version
        return out
    if interpretation.get('material_kind') != 'neuromatics':
        return interpretation
    if scoring_version != REFERENCE['scoring_version']:
        raise ValueError('Display classes belong to a different prototype scorer')
    out = deepcopy(interpretation)
    for task, card in [('Q', out['overall'])] + [(c['task'], c) for c in out['cards']]:
        reference = REFERENCE['references'].get(task)
        if reference is None:
            continue
        low, high = reference['cuts']
        card['level'] = ('Ниже типичного уровня' if card['score'] < low - 1e-12 else
                         'Выше типичного уровня' if card['score'] > high + 1e-12 else 'Типичный уровень')
    out['display_class_calibration'] = VERSION
    return out
