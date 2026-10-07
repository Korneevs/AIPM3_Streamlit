"""Plain-language feature definitions and corroborated video observations.

This module formats existing evidence. It neither measures features nor changes
scores; in particular, averaged scoring inputs are not treated as video facts.
"""
from __future__ import annotations

from collections.abc import Mapping
import re

from .latest_interpretation import has_person_name


FEATURE_MEANINGS = {
    "character_close_up_seconds": (
        "Суммарное время в секундах, когда лица персонажей показаны крупно - от плеч и выше. "
        "Учитываются все такие планы за ролик."
    ),
    "monologue_to_camera": (
        "Проверяем, обращается ли герой прямо к зрителю, глядя в камеру. "
        "Отмечается наличие такого обращения, а не его длительность."
    ),
    "pack_shot_duration_seconds": (
        "Длительность финального кадра, в котором на первом плане логотип, слоган или призыв к действию. "
        "Считается в секундах; если такого кадра нет, длительность равна нулю."
    ),
    "panel__first_core_claim_time_band": (
        "Смотрим, когда впервые становится явно понятно главное предложение: в первые 5 секунд, "
        "на 5-10-й, на 10-15-й или позже 15-й секунды. Отдельно отмечается случай, когда такого момента нет."
    ),
    "phys__audio_dynamic_range_db": (
        "Разница в громкости между тихими и громкими участками звука, измеренная в децибелах. "
        "Почти полная тишина и отдельные крайние всплески не определяют это значение."
    ),
    "phys__motion_mean": (
        "Насколько обычно меняется изображение между соседними кадрами при равномерном просмотре ролика. "
        "Это изменение картинки из-за движения или смены планов, а не число склеек или скорость героя."
    ),
    "product_demo_seconds": (
        "Суммарное время в секундах, когда показано использование продукта или работа сервиса - например, "
        "действие в интерфейсе. Само появление продукта без показа его работы не считается демонстрацией."
    ),
    "state_transformation_present": (
        "Проверяем, показан ли контраст «до и после»: было трудно или неудобно, а благодаря продукту стало лучше. "
        "Отмечается наличие такой перемены, а не просто смена настроения или сцены."
    ),
    "fresh__audiovisual_claim_alignment": (
        "Здесь смотрим, показано ли то действие или результат, о котором говорит озвучка. "
        "Различаем отсутствие связи, только общую тему или повтор слов на экране, показ действия либо результата и показ обоих вместе."
    ),
    "main_character": (
        "Проверяем, построена ли история вокруг одного главного героя. "
        "Другие люди могут быть в кадре - важно, есть ли у истории один центральный персонаж."
    ),
    "numeric_offer_on_screen": (
        "Проверяем, есть ли на экране конкретное предложение с ценой, процентом скидки или сроком. "
        "Также учитывается слово «бесплатно», если указано, что именно предлагается бесплатно."
    ),
    "panel__distinct_benefit_count": (
        "Считаем разные выгоды для пользователя, которые прямо названы или однозначно показаны. "
        "Повторы одной выгоды не добавляют новых; в оценке используется число от 0 до 4."
    ),
    "panel__mandatory_inference_chain_length": (
        "Считаем смысловые переходы, которые зрителю нужно самостоятельно додумать, чтобы связать события с главным предложением. "
        "Ноль означает прямую связь, дальше отмечается от одного до четырёх таких переходов."
    ),
    "panel__message_specificity_level": (
        "Смотрим, что именно раскрывает предложение: только бренд или настроение, общую выгоду, конкретный предмет или действие. "
        "Самый подробный вариант называет предмет, действие и результат либо условие."
    ),
    "panel__offer_novelty_explanation_need": (
        "Оцениваем, сколько пояснений требует непривычный продукт, функция или условие предложения - от почти никаких до значительных. "
        "Речь о сложности самого предложения, а не о количестве объяснений, уже прозвучавших в ролике."
    ),
    "panel__product_role_reveal_time_band": (
        "Смотрим, когда впервые понятно, какую роль сервис играет в истории: в первые 5 секунд, на 5-10-й, на 10-15-й или позже 15-й секунды. "
        "Отдельно отмечается случай, когда эта роль так и не объяснена."
    ),
    "promo": (
        "Проверяем, есть ли явная акция, скидка, бонус, промокод или подарок. "
        "Обычная выгода сервиса сама по себе акцией не считается."
    ),
    "brand_first_mention_seconds": (
        "Время в секундах до первого произнесённого названия Авито или его появления на экране. "
        "Если упоминания нет, в расчёт записывается длительность ролика - это не означает, что бренд появился в финале."
    ),
    "has_callback_to_opening": (
        "Проверяем, возвращается ли финал к ситуации или образу из первой сцены. "
        "Отмечается наличие этой связи между началом и концом истории."
    ),
    "human_characters_count": (
        "Считаем отдельных людей, которых можно различить как персонажей. "
        "Неразличимая толпа не добавляет персонажей; один выделенный герой на её фоне считается одним."
    ),
    "jingle_present": (
        "Проверяем, звучит ли короткая мелодическая фраза, связанная с брендом. "
        "Обычная фоновая музыка без такого мотива не считается фирменной мелодией."
    ),
    "problem_solution_arc_present": (
        "Проверяем, есть ли история «проблема - решение»: сначала проблема явно показана или названа, затем продукт её решает. "
        "Сам по себе показ продукта без исходной проблемы такой историей не считается."
    ),
    "scene_pace_high": (
        "Проверяем, меняются ли планы в среднем не реже одного раза в секунду. "
        "Отмечается высокий или невысокий темп монтажа, а не скорость движения внутри кадра."
    ),
}


_AUDIO_DEPENDENT = {
    "monologue_to_camera", "panel__first_core_claim_time_band",
    "phys__audio_dynamic_range_db", "fresh__audiovisual_claim_alignment",
    "panel__distinct_benefit_count", "panel__mandatory_inference_chain_length",
    "panel__message_specificity_level", "panel__offer_novelty_explanation_need",
    "panel__product_role_reveal_time_band", "brand_first_mention_seconds",
    "jingle_present", "problem_solution_arc_present",
}
_UNCONFIRMED = "Надёжно подтвердить эту деталь по имеющимся проверкам ролика не удалось."
_REQUIRES_REVIEW = "Описание этой детали требует проверки по ролику."
_INTERNAL_LANGUAGE = re.compile(
    r"\b(?:rating|score|shap|value|балл\w*|оценк\w*|уровень\s+\d|шкал\w*)\b", re.I
)
_AUDIENCE_CERTAINTY = re.compile(
    r"\b(?:понят\w*|яс\w*|очевид\w*)\s+(?:абсолютно\s+)?(?:всем|каждому|любому)\b"
    r"|\b(?:все|каждый|любой)\s+(?:зрител\w*\s+)?"
    r"(?:(?:обязательно|точно|сразу)\s+)?(?:пойм\w*|поня\w*|запом\w*|замет\w*)\b",
    re.I,
)


def _plain_text(value):
    if not isinstance(value, str):
        return ""
    text = value.replace("—", "-").replace("–", "-")
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"\bоффер(ов|ы|а|ом)?\b", lambda m: {
        None: "предложение", "ов": "предложений", "ы": "предложения",
        "а": "предложения", "ом": "предложением",
    }[m.group(1)], text, flags=re.I)
    text = re.sub(r"\bп[эе]кшот(е|а)?\b", lambda m: {
        None: "финальный кадр с брендом", "е": "финальном кадре с брендом",
        "а": "финального кадра с брендом",
    }[m.group(1)], text, flags=re.I)
    text = text.replace("Финальный финальный", "Финальный").replace("финальный финальный", "финальный")
    text = re.sub(r"([Фф]инальный) кадр с брендом с логотипом", r"\1 кадр с логотипом", text)
    return re.sub(r"\s+", " ", text).strip()


def _russian(text):
    return len(re.findall(r"[а-яё]", text, re.I)) > max(
        8, len(re.findall(r"[a-z]", text, re.I)))


def _brief_observation(text):
    """Keep an intact factual sentence, without exposing rubric scores."""
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        if (sentence and len(sentence) <= 360 and _russian(sentence)
                and not _INTERNAL_LANGUAGE.search(sentence)
                and not _AUDIENCE_CERTAINTY.search(sentence)):
            # Duration estimates in a corroborated description can still differ
            # between reads. Do not turn one read's timestamp into exact timing.
            timed = bool(re.search(
                r"\d+(?:[.,]\d+)?\s*(?:сек\w*|с\b)|\b\d{1,2}:\d{2}(?::\d{2})?\b",
                sentence, re.I))
            approximate = bool(re.search(r"примерно|около|приблизительно", sentence, re.I))
            if timed and not approximate:
                sentence = "По приблизительной оценке времени: " + sentence
            return sentence
    return ""


def _alignment_observation(support):
    """Structured flags confirm presence only; False never proves absence."""
    action = support.get("shown_action") is True
    result = support.get("shown_result") is True
    if action and result:
        return "В кадре подтверждены и действие, и результат из основной озвучки."
    if action:
        return "Действие из основной озвучки показано; его результат по этим кадрам подтвердить не удалось."
    if result:
        return "Результат из основной озвучки показан; действие, которое к нему приводит, по этим кадрам подтвердить не удалось."
    return "По имеющемуся описанию нельзя уверенно подтвердить показ действия или результата из основной озвучки."


def feature_observation(driver, audio_incomplete=False):
    """Return one supported observation, or an honest limitation, in Russian.

    ``driver`` is an existing interpretation driver. ``evidence.verified`` is
    required; a separately verified ``factual_observation`` is an explicitly
    neutral fallback when the evidence for the scoring input is unconfirmed.
    Neither ``value``, ``usable`` nor the contribution's sign creates a fact.
    """
    if not isinstance(driver, Mapping):
        return _UNCONFIRMED
    feature = driver.get("feature", "")
    if audio_incomplete and feature in _AUDIO_DEPENDENT:
        return "Без полной озвучки эту деталь сообщения нельзя уверенно проверить."

    support = driver.get("evidence")
    separate_fact = False
    if not isinstance(support, Mapping) or support.get("verified") is not True:
        fact = driver.get("factual_observation")
        support = fact.get("evidence") if isinstance(fact, Mapping) else None
        if not isinstance(support, Mapping) or support.get("verified") is not True:
            if feature == "human_characters_count":
                return "Число отдельных персонажей пока не подтверждено сверкой по кадрам."
            return _UNCONFIRMED
        separate_fact = True

    text = _plain_text(support.get("observation"))
    if has_person_name(text + " " + str(support.get("episodes", []))):
        return _REQUIRES_REVIEW
    if feature == "panel__offer_novelty_explanation_need" and re.search(
            r"\bне\s+требу(?:ет|ют|ется|ются)\s+(?:дополнительн\w*\s+)?объяснен\w*"
            r"|\b(?:дополнительн\w*\s+)?объяснен\w*\s+не\s+требу(?:ет|ют|ется|ются)",
            text, re.I):
        # A corroborated rubric assessment is not evidence of familiarity or
        # comprehension among viewers. State what the video reviews identified.
        text = "В проверках ролика не выделено сложных для объяснения условий предложения."
    observation = _brief_observation(text)
    if not observation and feature == "fresh__audiovisual_claim_alignment":
        observation = _alignment_observation(support)
    if not observation:
        return _REQUIRES_REVIEW
    return ("Отдельное наблюдение: " if separate_fact else "") + observation
