from karotte.schemas.transcript import ErrorEvent
from karotte.transcript_markdown import convert_error_event_to_markdown


def test_error_event_without_traceback():
    event = ErrorEvent(
        exception_type="ValueError",
        message="Something went wrong",
        traceback=None,
    )

    markdown = convert_error_event_to_markdown(event)

    assert "### 💥 Error" in markdown
    assert "**Exception Type:** `ValueError`" in markdown
    assert "**Message:**" in markdown
    assert "Something went wrong" in markdown
    assert "**Traceback:**" not in markdown


def test_error_event_with_traceback():
    traceback_text = """Traceback (most recent call last):
  File "test.py", line 10, in <module>
    raise ValueError("Something went wrong")
ValueError: Something went wrong"""

    event = ErrorEvent(
        exception_type="ValueError",
        message="Something went wrong",
        traceback=traceback_text,
    )

    markdown = convert_error_event_to_markdown(event)

    assert "### 💥 Error" in markdown
    assert "**Exception Type:** `ValueError`" in markdown
    assert "**Message:**" in markdown
    assert "Something went wrong" in markdown
    assert "**Traceback:**" in markdown
    assert 'File "test.py", line 10' in markdown
    assert "raise ValueError" in markdown
