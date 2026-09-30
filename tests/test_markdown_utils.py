from datetime import datetime
from typing import Any

import pytest

from karotte.transcript_markdown import add_code_block_to_markdown


def test_add_code_block_to_markdown_handles_string_single_line():
    assert add_code_block_to_markdown("Hello, world!") == "```\nHello, world!\n```\n\n"


def test_add_code_block_to_markdown_handles_string_multi_line():
    assert (
        add_code_block_to_markdown("Hello,\nworld!") == "```\nHello,\nworld!\n```\n\n"
    )


def test_add_code_block_to_markdown_handles_string_empty():
    assert add_code_block_to_markdown("") == "```\n```\n\n"


@pytest.mark.parametrize("value", [None, 1, 1.5, True, set()])
def test_add_code_block_to_markdown_handles_non_string_non_json_values(value: Any):
    assert add_code_block_to_markdown(value) == f"```\n{value}\n```\n\n"


class TestPrettyPrintDict:
    def test_empty_dict(self):
        assert add_code_block_to_markdown({}) == "```\n{}\n```\n\n"

    def test_simple_dict(self):
        result = add_code_block_to_markdown({"key": "value"})
        assert result == '```\n{\n  "key": "value"\n}\n```\n\n'

    def test_nested_dict(self):
        result = add_code_block_to_markdown({"outer": {"inner": "value"}})
        assert '"outer"' in result
        assert '"inner"' in result
        assert "  " in result  # indentation

    def test_dict_with_multiple_keys(self):
        result = add_code_block_to_markdown({"a": 1, "b": 2})
        assert '"a": 1' in result
        assert '"b": 2' in result

    def test_non_serializable_dict_falls_back_to_str(self):
        non_serializable = {"date": datetime(2024, 1, 1)}
        result = add_code_block_to_markdown(non_serializable)
        assert "```\n" in result
        assert "datetime" in result


class TestPrettyPrintList:
    def test_empty_list(self):
        assert add_code_block_to_markdown([]) == "```\n[]\n```\n\n"

    def test_simple_list(self):
        result = add_code_block_to_markdown([1, 2, 3])
        assert result == "```\n[\n  1,\n  2,\n  3\n]\n```\n\n"

    def test_list_of_dicts(self):
        result = add_code_block_to_markdown([{"a": 1}, {"b": 2}])
        assert '"a": 1' in result
        assert '"b": 2' in result

    def test_non_serializable_list_falls_back_to_str(self):
        non_serializable = [datetime(2024, 1, 1)]
        result = add_code_block_to_markdown(non_serializable)
        assert "```\n" in result
        assert "datetime" in result


class TestLanguageParameter:
    def test_with_language(self):
        result = add_code_block_to_markdown("code", language="python")
        assert result == "```python\ncode\n```\n\n"

    def test_with_json_language(self):
        result = add_code_block_to_markdown({"key": "value"}, language="json")
        assert result.startswith("```json\n")
