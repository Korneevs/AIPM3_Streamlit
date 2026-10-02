import pytest
from aipm3.display_calibration import audio_status, display_values, neuromatics_reference
from aipm3.latest_runtime import NEUROMATICS_SCORING_VERSION


def test_display_reference_rejects_a_different_model():
    with pytest.raises(ValueError): neuromatics_reference('another-model')
    assert neuromatics_reference(NEUROMATICS_SCORING_VERSION)['pair_count'] == 12


def test_audio_metadata_is_bound_to_the_actual_source_not_filename():
    assert audio_status(dict(source_sha='first',audio_review=dict(source_sha='second',status='complete'))) == 'unknown'
    assert audio_status(dict(source_sha='first',audio_review=dict(source_sha='first',status='partial'))) == 'partial'
    assert audio_status({}) == 'unknown'


def test_exact_class_boundaries_remain_in_middle_class():
    ref=dict(mean=2,cuts=[1,3])
    assert display_values(.9,ref)['level']=='Ниже типичного уровня'
    assert display_values(1,ref)['level']=='Типичный уровень'
    assert display_values(3,ref)['level']=='Типичный уровень'
    assert display_values(3.1,ref)['level']=='Выше типичного уровня'
