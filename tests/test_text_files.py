import pytest

from karotte.text_files import decode_text


def test_decodes_utf8():
    assert decode_text("café".encode(), "f") == "café"


def test_rejects_non_utf8():
    with pytest.raises(ValueError, match="Not a UTF-8 text file: f"):
        decode_text(b"\xff\xfe", "f")


def test_a_cut_off_last_character_is_forgiven_only_when_asked():
    cut = "é".encode()[:1]
    assert decode_text(cut, "f", complete=False) == ""
    with pytest.raises(ValueError, match="Not a UTF-8"):
        decode_text(cut, "f")


def test_incomplete_mode_still_rejects_garbage_in_the_middle():
    with pytest.raises(ValueError, match="Not a UTF-8"):
        decode_text(b"a\xffb", "f", complete=False)
