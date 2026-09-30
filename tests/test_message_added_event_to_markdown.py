from typing import Any

from karotte.schemas.chat import Message, Role
from karotte.schemas.transcript import MessageAddedEvent
from karotte.transcript_markdown import convert_message_added_event_to_markdown


def _render(content: str | list[dict[str, Any]], role: Role = "user") -> str:
    return convert_message_added_event_to_markdown(
        MessageAddedEvent(message=Message(role=role, content=content))
    )


def test_string_content() -> None:
    assert _render("hello\nworld") == "#### 📋 Instructions\n\nhello\nworld\n"


def test_content_parts_render_text_and_image_placeholder() -> None:
    result = _render(
        [
            {"type": "text", "text": "look at this"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
        ]
    )
    assert result == "#### 📋 Instructions\n\nlook at this\n\n[Image: image/png]\n\n"
    assert "TODO" not in result


def test_unknown_content_part_shows_type() -> None:
    assert _render([{"type": "input_audio"}]).endswith("[input_audio]\n\n")


def test_unknown_role_heading_is_separated_from_content() -> None:
    assert _render("hi", role="function") == "#### Function\n\nhi\n"
