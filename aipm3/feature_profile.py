"""Descriptive measurements, not new quality scores or feature attributions."""
from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
import json
import math
from pathlib import Path
from statistics import mean

VERSION = 'creative-profile-v1'
# Native rubric units are retained. Different constructs are never averaged.
CANDIDATES = {
    'main_character': ('story', 'Главный герой', 'aipm1', 1, 'binary'),
    'state_transformation': ('story', 'Изменение «до / после»', 'aipm1', 1, 'binary'),
    'humor': ('story', 'Юмор', 'aipm1', 1, 'binary'),
    'brand_logo_screen_seconds': ('brand', 'Логотип в кадре', 'aipm2', 100, 'duration'),
    'pack_shot_duration_seconds': ('brand', 'Финальный пэкшот', 'aipm2', 100, 'duration'),
    'core_claim_quartile_coverage': ('structure', 'Посыл в частях ролика', 'panel', 4, 'ordinal'),
    'offer_condition_count': ('structure', 'Условия предложения', 'panel', 4, 'ordinal'),
    'audio_only_message_completeness': ('channels', 'Только по звуку', 'panel', 3, 'ordinal'),
    'visual_only_message_completeness': ('channels', 'Без звука', 'panel', 3, 'ordinal'),
    # Audited alternatives are retained for a reproducible selection record.
    'cta_clarity': ('alternatives', 'Ясность следующего действия', 'panel', 1, 'ordinal'),
    'message_specificity_level': ('alternatives', 'Конкретность сообщения', 'panel', 3, 'ordinal'),
}
GROUPS = {
    'story': ('Сюжетные опоры', 'Какие приёмы присутствуют — без оценки «чем больше, тем лучше».'),
    'brand': ('Присутствие бренда', 'Доля хронометража; не складываем пересекающиеся интервалы.'),
    'structure': ('Структура сообщения', 'Где повторяется посыл и сколько условий нужно учесть.'),
    'channels': ('Передача сообщения', 'Средняя оценка полноты: 0 — посыл не определить, 3 — полный пересказ.'),
}
MAX_REPEAT_DELTA = 0.10  # Engineering tolerance, fraction of native scale; not a statistical margin.


def _number(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (ValueError, TypeError):
        return None


def _values(result, feature, source):
    if source in {'aipm1', 'aipm2'}:
        runs = result.get('objective_runs', {})
        rows = runs.get(source, []) if isinstance(runs, dict) else []
        expected = 3 if source == 'aipm1' else 2
        if not isinstance(rows, list) or len(rows) != expected:
            return None
        values = [_number(r.get(feature)) for r in rows if isinstance(r, dict)]
        return values if len(values) == expected and all(v is not None for v in values) else None
    rows = result.get('diagnostic_panel', [])
    if not isinstance(rows, list) or len(rows) != 30:
        return None
    by_call = defaultdict(list)
    ids = []
    for r in rows:
        if not isinstance(r, dict):
            return None
        rid, call = r.get('respondent_id'), r.get('call_id')
        if not isinstance(rid, str) or type(call) is not int or call not in range(1, 11):
            return None
        if rid not in {f'p{n:02d}' for n in range(3 * (call - 1) + 1, 3 * call + 1)}:
            return None
        value = _number(r.get(feature))
        if value is None or value < 0 or value > CANDIDATES[feature][3] or not value.is_integer():
            return None
        ids.append(rid)
        by_call[call].append(value)
    if len(set(ids)) != 30 or len(by_call) != 10 or any(len(v) != 3 for v in by_call.values()):
        return None
    # Persona answers within one call are not independent observations.
    return [mean(by_call[call]) for call in sorted(by_call)]


def measure_profile(result):
    """No API calls, scoring calls, model effects, train ranks or state writes."""
    measured = []
    for feature, (group, label, source, maximum, kind) in CANDIDATES.items():
        values = _values(result, feature, source)
        item = {'feature': feature, 'group': group, 'label': label, 'source': source,
                'maximum': maximum, 'kind': kind, 'value': None, 'within_stable': False}
        duration = _number(result.get('duration_seconds'))
        if values is None or any(v < 0 for v in values):
            measured.append(item)
            continue
        native_values = values
        if kind == 'duration':
            # Legacy seconds are integer-rounded. Only this rounding excess is allowed.
            if not duration or duration <= 0 or any(v > math.ceil(duration) for v in values):
                measured.append(item)
                continue
            values = [min(100., 100 * v / duration) for v in values]
        elif any(v > maximum for v in values):
            measured.append(item)
            continue
        elif kind == 'binary' and any(v not in {0., 1.} for v in values):
            measured.append(item)
            continue
        value, low, high = mean(values), min(values), max(values)
        # Different persona groups are not repeat sessions. Do not convert
        # within-panel dispersion into a claim of between-run stability.
        stable = low == high if kind == 'binary' else None
        item.update(value=value, low=low, high=high, call_values=values, calls=len(values),
                    native_value=mean(native_values), within_stable=stable,
                    display=(('Есть' if value == 1 else 'Нет') if kind == 'binary' and stable else
                             'Неоднозначно' if kind == 'binary' else
                             f'≈{mean(native_values):g} сек. · {value:.0f}%' if kind == 'duration' else
                             f'{value:.1f} / {maximum}'))
        measured.append(item)
    return measured


@lru_cache(maxsize=1)
def validation_manifest():
    return json.loads(Path(__file__).with_name('profile_validation.json').read_text())


def build_profile(result):
    validation = validation_manifest()
    if validation.get('profile_version') != VERSION:
        raise ValueError('Profile validation version mismatch')
    approved = set(validation['approved_features'])
    visible, withheld = [], []
    for item in measure_profile(result):
        if item['group'] not in GROUPS:
            continue
        if item['feature'] not in approved:
            item['reason'] = 'Не прошёл межзапусковую проверку'
            withheld.append(item)
            continue
        if item['value'] is None:
            item['reason'] = 'Нет полного набора исходных разметок'
        elif item['kind'] == 'binary' and not item['within_stable']:
            item['reason'] = 'Разметки текущего запуска расходятся'
        else:
            visible.append(item)
            continue
        withheld.append(item)
    return {'version': VERSION, 'measurements': visible, 'withheld': withheld,
            'validation': validation}
