"""Presentation-only classes, calibrated for the fixed-mean prototype scorer."""
from copy import deepcopy
import json
from pathlib import Path

REFERENCE = json.loads((Path(__file__).with_name('manager_display_classes.json')).read_text())
VERSION = REFERENCE['version']


def with_display_classes(interpretation, scoring_version):
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
