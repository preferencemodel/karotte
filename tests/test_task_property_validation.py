"""`Task.__init_subclass__` rejects property fields overridden with a bare `def`.

Forgetting the `@property` on an override of `tools`/`required_hardware`/etc. is
silently accepted by ABCMeta, after which the attribute resolves to a bound
method and fails far from the typo (e.g. `karotte tasks list` dies with "Object of
type method is not JSON serializable"). The guard turns that into a clear error
at class-definition time.

Subclasses here are built with `type()` rather than `class` statements: the
"bad" cases are deliberately type-incorrect (a method where a property is
declared), so a static `class` block would only trip the type checker. The
runtime guard fires identically either way.
"""

from typing import Any

import pytest

from karotte import Task


def _steps(_self: Any) -> list[Any]:
    return []


def _tools(_self: Any) -> list[str]:
    return ["bash"]


def _required_hardware(_self: Any) -> str:
    return "small"


def test_required_hardware_without_property_raises() -> None:
    with pytest.raises(TypeError, match="required_hardware"):
        type(
            "BadHardwareTask",
            (Task,),
            {
                "id": "bad-hardware",
                "steps": property(_steps),
                "tools": property(_tools),
                # Missing @property — the exact regression this guards against.
                "required_hardware": _required_hardware,
            },
        )


def test_tools_without_property_raises() -> None:
    with pytest.raises(TypeError, match="tools"):
        type(
            "BadToolsTask",
            (Task,),
            {
                "id": "bad-tools",
                "steps": property(_steps),
                # Missing @property.
                "tools": _tools,
            },
        )


def test_correctly_decorated_task_is_accepted() -> None:
    cls = type(
        "GoodTask",
        (Task,),
        {
            "id": "good-task",
            "steps": property(_steps),
            "tools": property(_tools),
            "required_hardware": property(_required_hardware),
        },
    )
    assert isinstance(cls.__dict__["required_hardware"], property)


def test_class_attribute_override_is_accepted() -> None:
    """A plain value (not a method) is fine — only a forgotten @property is rejected."""
    cls = type(
        "ClassAttrTask",
        (Task,),
        {
            "id": "class-attr-task",
            "steps": property(_steps),
            "tools": property(_tools),
            "required_hardware": "small",
        },
    )
    assert cls.__dict__["required_hardware"] == "small"


def _scoring_limit(_self: Any) -> float:
    return 60.0


def test_scoring_time_limit_without_property_raises() -> None:
    with pytest.raises(TypeError, match="scoring_time_limit_seconds"):
        type(
            "BadScoringLimitTask",
            (Task,),
            {
                "id": "bad-scoring-limit",
                "system_prompt": property(lambda _self: None),
                "steps": property(_steps),
                "tools": property(_tools),
                "scoring_time_limit_seconds": _scoring_limit,
            },
        )
