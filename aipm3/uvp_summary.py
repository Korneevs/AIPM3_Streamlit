"""Select the strongest positive UVPs and images without changing model scores."""
from __future__ import annotations

from collections import OrderedDict
from typing import Any


POSITIVE_LABELS = {3: "Сильно выражено", 2: "Умеренно выражено", 1: "Слабо выражено"}
GENERAL_IMAGES = "Общие имиджи"


def build_uvp_summary(result: dict[str, Any]) -> dict[str, Any]:
    """Summarize UVPs independently of general brand images.

    UVPs inherit the highest positive score among their associated images,
    including secondary associations. Only the strongest positive UVP tier is
    selected. Main images are selected independently across all image groups.
    The review flag is deliberately irrelevant to both selections.
    """
    images = result.get("images", [])
    positive = [image for image in images if image["score"] > 0]
    image_score = max((image["score"] for image in positive), default=None)
    main_images = [image for image in positive if image["score"] == image_score]

    groups: OrderedDict[str, dict[str, Any]] = OrderedDict()
    for image in positive:
        names = [image["group"], *image.get("secondary_groups", [])]
        for name in dict.fromkeys(names):
            if not name or name == GENERAL_IMAGES:
                continue
            group = groups.setdefault(name, {"label": name, "score": image["score"], "images": []})
            group["score"] = max(group["score"], image["score"])
            group["images"].append(image)

    uvp_score = max((group["score"] for group in groups.values()), default=None)
    main_uvps = [
        {**group, "images": [image for image in group["images"] if image["score"] == group["score"]]}
        for group in groups.values()
        if group["score"] == uvp_score
    ]
    return {"uvps": main_uvps, "uvp_score": uvp_score,
            "images": main_images, "image_score": image_score}
