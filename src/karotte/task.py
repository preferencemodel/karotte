from abc import ABC, ABCMeta, abstractmethod
from collections.abc import Iterable
from pathlib import Path
from types import FunctionType
from typing import Any, Final

from karotte.hardware import default_hardware
from karotte.schemas import EvaluationRunConfig
from karotte.schemas.data_mount import DataMount
from karotte.step import Step

_MISSING_SYSTEM_PROMPT_HELP = """{cls} does not define a 'system_prompt' property.

Environments based on the default template define it like this:

    from environment.system_prompts import get_system_prompt

    @property
    def system_prompt(self) -> str:
        return get_system_prompt(self.config.model, self.config.extra_config)

Return None instead if the task should run without a system message."""


class _TaskMeta(ABCMeta):
    """Replaces the bare ABC error for a missing `system_prompt` with one that
    says what to write."""

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        if "system_prompt" in getattr(cls, "__abstractmethods__", frozenset()):
            raise TypeError(
                _MISSING_SYSTEM_PROMPT_HELP.format(
                    cls=f"{cls.__module__}.{cls.__qualname__}"
                )
            )
        return super().__call__(*args, **kwargs)


class Task(ABC, metaclass=_TaskMeta):
    """Receives the run configuration during initialization.

    This allows making task properties depend on the run configuration.
    """

    id: str

    def __init__(self, config: EvaluationRunConfig):
        self.config: Final = config

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        # `system_prompt`, `steps`, `tools`, `required_hardware` and `data_mounts`
        # are declared as properties here and read as plain values elsewhere (e.g.
        # `karotte tasks list` serializes them to JSON). Overriding one with a bare
        # `def` — a forgotten @property — is silently accepted by ABCMeta, after
        # which `task.<field>` is a bound method that fails far from the typo with
        # an opaque "Object of type method is not JSON serializable". Reject it at
        # class-definition time with a message that names the offending field.
        property_fields = {
            name for name, value in vars(Task).items() if isinstance(value, property)
        }
        for name in sorted(property_fields):
            if isinstance(cls.__dict__.get(name), FunctionType):
                raise TypeError(
                    f"{cls.__module__}.{cls.__qualname__} overrides '{name}' with a method, "
                    + "but Task declares it as a property. Add the @property decorator — "
                    + f"otherwise '{name}' resolves to a bound method, which breaks task "
                    + "loading and JSON serialization."
                )

    @property
    @abstractmethod
    def system_prompt(self) -> str | None:
        """The system prompt sent to the model at the start of the evaluation.

        Return `None` to start the run without a system message.
        """

    def configure_tools(self) -> None:
        """Gets called before the MCP server gets initialized.

        Use this to configure any tool settings required for this task.
        """
        pass

    def pre_hook(self) -> dict[str, Any]:
        """Gets called before the task gets executed.

        Use this to do any dynamic setup for the task.
        """
        return {}

    @property
    @abstractmethod
    def steps(self) -> Iterable[Step]: ...

    @property
    @abstractmethod
    def tools(self) -> list[str]: ...

    @property
    def required_hardware(self) -> str | None:
        """The hardware this task runs on, named as an installed plugin knows it.

        Defaults to the plugin's default, or ``None`` without a plugin.
        """
        return default_hardware()

    @property
    def submission_paths(self) -> tuple[Path, ...]:
        """Every path the student hands in, collected from the task's steps."""
        return tuple(
            dict.fromkeys(
                path for step in self.steps for path in (step.submission_paths or ())
            )
        )

    @property
    def data_mounts(self) -> list[DataMount]:
        """Data mounts to attach from the backend's mount registry at runtime."""
        return []
