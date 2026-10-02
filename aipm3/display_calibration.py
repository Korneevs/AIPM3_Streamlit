"""Display references only. Never used for raw scores or creative coefficients."""
import json
from pathlib import Path


def neuromatics_reference(scoring_version):
    reference = json.loads(Path(__file__).with_name('neuromatics_display_reference.json').read_text())
    if reference['scoring_version'] != scoring_version:
        raise ValueError('Display calibration belongs to another neuromatics model')
    return reference


def display_values(score, reference):
    low, high = reference['cuts']
    return dict(index=100 * score / reference['mean'], baseline=reference['mean'],
                level=('Выше типичного уровня' if score > high else
                       'Ниже типичного уровня' if score < low else 'Типичный уровень'))


def audio_status(result):
    """Audio presence cannot establish completeness. Accept only source-bound reviews."""
    review = result.get('audio_review') or {}
    if review.get('source_sha') != result.get('source_sha'):
        return 'unknown'
    status = review.get('status')
    return status if status in {'complete', 'partial', 'absent'} else 'unknown'


def audio_note(status):
    if status == 'partial':
        return ('Полная озвучка этой версии не подтверждена. Ниже разобраны доступные детали. '
                'Сравнивать общую оценку с нормой пока нельзя: недостающая речь может изменить смысл и связь истории с брендом.')
    if status == 'absent':
        return ('В этой версии нет озвучки. Ниже разобраны доступные детали. '
                'Сравнивать общую оценку с нормой пока нельзя: звук может изменить смысл и связь истории с брендом.')
    if status == 'unknown':
        return 'Полнота озвучки не подтверждена. Если в файле есть не вся речь, сравнение с нормой пока ненадёжно.'
    return ''
