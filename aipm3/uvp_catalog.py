"""BUTVA image questions by vertical. Wording variants share one score."""
import json
from functools import lru_cache
from pathlib import Path

VERTICALS = ("Товары", "Услуги", "Путешествия", "Недвижимость", "Авто", "Работа", "Горизонт")


@lru_cache(maxsize=1)
def _catalog():
    return json.loads(Path(__file__).with_name("uvp_image_catalog.json").read_text(encoding="utf-8"))


def questions_for(vertical):
    if vertical not in VERTICALS:
        raise ValueError("Неизвестная вертикаль")
    questions = []
    for image_id, card in _catalog().items():
        variants = card["variants"].get(vertical)
        if variants:
            questions.append({
                "id": image_id,
                "statement": variants[0] if len(variants) == 1 else card["statement"],
                "variants": list(variants),
                "group": card["group"],
                "secondary_groups": list(card.get("secondary_groups", [])),
            })
    return questions
