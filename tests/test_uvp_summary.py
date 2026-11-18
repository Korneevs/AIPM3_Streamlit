"""Business selection rules for independently summarized UVPs and images."""
from copy import deepcopy

import pytest

from aipm3.uvp_summary import build_uvp_summary


def image(label, group, score, *, secondary=(), review=False):
    return {"id": label, "label": label, "group": group, "score": score,
            "secondary_groups": list(secondary), "review": review}


def test_general_image_three_does_not_hide_profile_uvp_two():
    result = {"images": [
        image("Хорошие истории", "Общие имиджи", 3),
        image("Можно доверять", "Доверие", 2, review=True),
        image("Большой выбор", "Ассортимент", 1),
    ]}
    saved = deepcopy(result)
    summary = build_uvp_summary(result)
    assert summary["image_score"] == 3
    assert [item["label"] for item in summary["images"]] == ["Хорошие истории"]
    assert summary["uvp_score"] == 2
    assert [item["label"] for item in summary["uvps"]] == ["Доверие"]
    assert summary["uvps"][0]["images"][0]["review"] is True
    assert result == saved


def test_secondary_groups_contribute_without_duplicate_images_or_general_uvp():
    result = {"images": [
        image("Легко найти", "Подбор под запрос", 3,
              secondary=("Удобство и скорость", "Подбор под запрос", "Общие имиджи")),
        image("Не нужно ждать", "Удобство и скорость", 2),
        image("Не доверяю", "Доверие", -3),
    ]}
    summary = build_uvp_summary(result)
    assert [item["label"] for item in summary["uvps"]] == ["Подбор под запрос", "Удобство и скорость"]
    assert all(item["score"] == 3 for item in summary["uvps"])
    assert all([i["label"] for i in item["images"]] == ["Легко найти"] for item in summary["uvps"])


@pytest.mark.parametrize("scores", [[], [0], [-3, -1], [-3, 0]])
def test_no_positive_score_means_no_main_uvps_or_images(scores):
    result = {"images": [image(str(index), "Доверие", score) for index, score in enumerate(scores)]}
    summary = build_uvp_summary(result)
    assert summary == {"uvps": [], "uvp_score": None, "images": [], "image_score": None}


def test_weak_positive_is_labeled_as_one_and_review_does_not_change_selection():
    result = {"images": [image("Есть выбор", "Ассортимент", 1, review=True)]}
    summary = build_uvp_summary(result)
    assert summary["uvp_score"] == summary["image_score"] == 1
    assert summary["uvps"][0]["label"] == "Ассортимент"
    result["images"][0]["review"] = False
    other = build_uvp_summary(result)
    assert other["uvp_score"] == summary["uvp_score"]
    assert [item["label"] for item in other["uvps"]] == [item["label"] for item in summary["uvps"]]
