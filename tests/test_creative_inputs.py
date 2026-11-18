"""User text survives batch imports; malformed/oversized inputs are explicit."""
from io import BytesIO
from zipfile import ZipFile

import pytest

from aipm3.creative_inputs import CreativeInputError, collect_scenarios, parse_upload, safe_csv_cell


def test_thirty_plus_delimited_scripts_keep_all_text_and_titles():
    pasted = "\n\n".join(f"=== Вариант {i} ===\nРеплика {i}.\nКадр {i}." for i in range(35))
    rows = collect_scenarios([], pasted)
    assert len(rows) == 35
    assert len({row["id"] for row in rows}) == 35
    assert rows[34]["title"] == "Вариант 34"
    assert rows[34]["text"] == "Реплика 34.\nКадр 34."


def test_multiline_csv_and_combined_uploads_preserve_source_ids():
    source = 'id;title;text\ns001;Первый;"Реплика один.\nРеплика два."\n'.encode("utf-8-sig")
    rows = collect_scenarios([("batch.csv", source), ("Второй.txt", "Текст второго".encode())], "=== Третий ===\nТретий текст")
    assert len(rows) == 3 and len({row["id"] for row in rows}) == 3
    assert rows[0] == {"id": "s001", "title": "Первый", "text": "Реплика один.\nРеплика два."}
    assert rows[1]["title"] == "Второй" and rows[2]["title"] == "Третий"


def test_duplicate_ids_and_missing_bodies_are_not_silently_dropped():
    with pytest.raises(CreativeInputError, match="повторяются ID"):
        collect_scenarios([("a.csv", b"id,title,text\nx,One,Text\nx,Two,Text2\n")], "")
    with pytest.raises(CreativeInputError, match="нет текста"):
        collect_scenarios([], "=== Один ===\nТекст\n=== Два ===\n")
    with pytest.raises(CreativeInputError, match="Перед первым"):
        collect_scenarios([], "Потерявшийся текст\n=== Один ===\nТекст")
    with pytest.raises(CreativeInputError, match="полный текст"):
        collect_scenarios([("a.csv", b"title,text\nEmpty,\n")], "")


@pytest.mark.parametrize("header", ["title,text,text", "title,text, Text ", "title,text,Текст"])
def test_duplicate_csv_headers_are_rejected_before_any_text_is_lost(header):
    source = f"{header}\nНазвание,Первый важный текст,Второй текст\n".encode()
    with pytest.raises(CreativeInputError, match="повторяются столбцы"):
        collect_scenarios([("batch.csv", source)], "")


def test_unclosed_csv_quote_cannot_silently_merge_scenarios():
    source = 'title,text\nПервый,"Первый сценарий\nВторой,Второй сценарий\n'.encode()
    with pytest.raises(CreativeInputError, match="кавычки"):
        collect_scenarios([("batch.csv", source)], "")


def test_oversized_script_and_batch_are_rejected_without_truncation():
    with pytest.raises(CreativeInputError, match="длиннее"):
        collect_scenarios([], "x" * 12001)
    with pytest.raises(CreativeInputError, match="до 60"):
        collect_scenarios([], "\n".join(f"=== {i} ===\nText" for i in range(61)))
    with pytest.raises(CreativeInputError, match="Общий объём"):
        collect_scenarios([], "\n".join(f"=== {i} ===\n" + "x" * 10000 for i in range(19)))


def test_docx_text_in_tables_and_line_breaks_is_imported():
    source = BytesIO()
    with ZipFile(source, "w") as archive:
        archive.writestr("word/document.xml", '''<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>
        <w:p><w:r><w:t>Речь</w:t><w:br/><w:t>продолжается.</w:t></w:r></w:p>
        <w:tbl><w:tr><w:tc><w:p><w:r><w:t>Кадр в таблице.</w:t></w:r></w:p></w:tc></w:tr></w:tbl>
        </w:body></w:document>''')
    rows = parse_upload("Сценарий.docx", source.getvalue())
    assert rows == [{"title": "Сценарий", "text": "Речь\nпродолжается.\nКадр в таблице."}]


def test_invalid_encoding_and_broken_docx_are_clear_input_errors():
    with pytest.raises(CreativeInputError, match="UTF-8"):
        parse_upload("bad.txt", b"\xff\xfe")
    with pytest.raises(CreativeInputError, match="DOCX"):
        parse_upload("bad.docx", b"not a zip")


@pytest.mark.parametrize("text", ["=HYPERLINK(1)", "  +SUM(A1)", "@formula", "\ttext", "\rtext", "\ntext", "-10+1"])
def test_csv_export_neutralizes_formula_text(text):
    assert safe_csv_cell(text) == "'" + text


def test_csv_export_keeps_regular_text():
    assert safe_csv_cell("Сценарий") == "Сценарий"
