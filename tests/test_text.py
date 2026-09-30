from karotte.text import escape_surrogates, escape_surrogates_deep


def test_leaves_plain_text_alone():
    assert escape_surrogates("all good — é 漢字") == "all good — é 漢字"


def test_escapes_lone_surrogates():
    name = b"bad\xff\xfelink".decode("utf-8", "surrogateescape")

    escaped = escape_surrogates(f"{name} is a symlink")

    assert escaped == r"bad\udcff\udcfelink is a symlink"
    _ = escaped.encode("utf-8")


def test_escaping_is_idempotent():
    once = escape_surrogates(chr(0xDCFF))

    assert escape_surrogates(once) == once


def test_escapes_strings_nested_in_metadata():
    bad = chr(0xDCFF)

    escaped = escape_surrogates_deep(
        {"files": [bad, {bad: bad}], "count": 2, "ok": None}
    )

    assert escaped == {
        "files": [r"\udcff", {r"\udcff": r"\udcff"}],
        "count": 2,
        "ok": None,
    }
