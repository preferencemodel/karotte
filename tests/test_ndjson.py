import pytest

from karotte.agents.ndjson import parse_json_line


def test_an_object():
    assert parse_json_line('{"a": 1}', 0, "cli") == {"a": 1}


@pytest.mark.parametrize("line", ["", "not json", "[1]", "[" * 200_000])
def test_anything_else_is_skipped(line: str):
    assert parse_json_line(line, 0, "cli") is None
