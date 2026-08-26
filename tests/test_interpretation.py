from aipm3.interpretation import _top_level_summary


def test_high_delivery_low_noticeability_calls_out_noticeability() -> None:
    text = _top_level_summary("Низкий", "Средний", "Высокий")
    assert "заметность" in text.lower()


def test_high_noticeability_low_delivery_calls_out_understanding() -> None:
    text = _top_level_summary("Высокий", "Средний", "Низкий")
    assert "понимание" in text.lower()


def test_high_memorability_low_delivery_separates_memory_from_idea() -> None:
    text = _top_level_summary("Средний", "Высокий", "Низкий")
    assert "запомниться" in text.lower()
    assert "главная идея" in text.lower()
