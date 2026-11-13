import subprocess

import imageio_ffmpeg

from aipm3.uvp_catalog import VERTICALS, questions_for
from aipm3.uvp_media import prepare_evidence


def test_vertical_questions_are_distinct_and_contain_only_descriptive_fields():
    expected = {"Товары": 21, "Услуги": 8, "Путешествия": 15, "Недвижимость": 9,
                "Авто": 11, "Работа": 9, "Горизонт": 7}
    assert set(VERTICALS) == set(expected)
    seen = set()
    for vertical, count in expected.items():
        questions = questions_for(vertical)
        ids = {row["id"] for row in questions}
        assert len(questions) == len(ids) == count
        for row in questions:
            assert set(row) == {"id", "statement", "variants", "group", "secondary_groups"}
            assert row["variants"] and row["group"] and row["statement"]
        seen.update(ids)
    assert len(seen) == 52


def test_silent_vertical_clip_uses_its_entire_duration_without_external_calls(tmp_path):
    clip = tmp_path / "silent.mov"
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-f", "lavfi", "-i",
                    "color=c=green:s=90x160:r=25", "-t", "4", "-an", "-c:v", "libx264",
                    "-threads", "1", str(clip)], check=True)
    evidence = prepare_evidence(clip, api_base="", api_key="")
    assert evidence["speech"] == []
    assert len(evidence["sheets"]) == 5
    assert set(evidence["frame_times"]) == {f"V{i:02d}" for i in range(1, 41)}
    assert 0 <= evidence["frame_times"]["V01"] < .1 and 3.9 < evidence["frame_times"]["V40"] < 4
    assert all(s.startswith("data:image/jpeg;base64,") for s in evidence["sheets"])
