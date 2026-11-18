import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import inspect
import pkgutil
import sys
import types
import typing
from collections.abc import Callable, Sequence
from typing import Any

from fastmcp.tools import ToolResult
from loguru import logger

_FASTMCP3_TOOL_MODULE = "fastmcp.tools.tool"


class _Fastmcp3ToolModule(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Serves `fastmcp.tools.tool`, which fastmcp 4 removed, from `fastmcp.tools`.

    Tools written for fastmcp 3 import `ToolResult` from there.
    """

    def find_spec(
        self,
        fullname: str,
        path: Sequence[str] | None,
        target: types.ModuleType | None = None,
    ) -> importlib.machinery.ModuleSpec | None:
        if fullname != _FASTMCP3_TOOL_MODULE:
            return None
        return importlib.util.spec_from_loader(fullname, self)

    def exec_module(self, module: types.ModuleType) -> None:
        logger.warning(
            f"`{_FASTMCP3_TOOL_MODULE}` was removed in fastmcp 4; "
            + "import from `fastmcp.tools` instead"
        )
        import fastmcp.tools
        import fastmcp.tools.base
        import fastmcp.tools.function_tool

        def __getattr__(name: str) -> Any:
            for source in (fastmcp.tools.base, fastmcp.tools.function_tool):
                if hasattr(source, name):
                    return getattr(source, name)
            raise AttributeError(
                f"module {module.__name__!r} has no attribute {name!r}"
            )

        module.__dict__.update(
            {name: getattr(fastmcp.tools, name) for name in fastmcp.tools.__all__}
        )
        module.__getattr__ = __getattr__  # type: ignore[method-assign]


if not any(isinstance(f, _Fastmcp3ToolModule) for f in sys.meta_path):
    sys.meta_path.append(_Fastmcp3ToolModule())


class InvalidToolError(Exception):
    """Raised when a tool fails validation."""


def discover_tools(tool_names: list[str] | None = None) -> dict[str, object]:
    """Returns validated tool modules from ``karotte.tools`` and ``environment.tools``.

    Args:
        tool_names: If provided, validate and return only these tools.
            If None, discovers and validates all tools.

    Returns:
        Mapping of tool name to its already-imported module.

    Raises:
        InvalidToolError: If any tool is not found or fails validation.
    """
    candidates = _collect_all_candidates()

    if tool_names is not None:
        candidates = _filter_candidates(candidates, tool_names)

    for tool_name, module in candidates.items():
        _validate_tool(module, tool_name)

    return candidates


def _collect_all_candidates() -> dict[str, object]:
    """Collect all candidate modules from karotte.tools and environment.tools (if it exists)."""
    import karotte.tools

    paths_to_search: dict[str, Sequence[str]] = {
        "karotte.tools": karotte.tools.__path__
    }

    try:
        import environment.tools  # pyright: ignore[reportMissingImports]

        paths_to_search["environment.tools"] = environment.tools.__path__
    except ModuleNotFoundError:
        pass

    candidates: dict[str, object] = {}
    for prefix, path in paths_to_search.items():
        for module_info in pkgutil.iter_modules(path):
            module_name = module_info.name
            try:
                module = importlib.import_module(f"{prefix}.{module_name}")
                candidates[module_name] = module
            except ModuleNotFoundError:
                pass

    return candidates


def _filter_candidates(
    candidates: dict[str, object], tool_names: list[str]
) -> dict[str, object]:
    """Filter candidates by tool_names. Raises if any requested tool is missing."""
    filtered: dict[str, object] = {}
    for tool_name in tool_names:
        if tool_name not in candidates:
            raise InvalidToolError(f"Tool {tool_name!r} not found")
        filtered[tool_name] = candidates[tool_name]
    return filtered


def _validate_tool(module: object, tool_name: str) -> None:
    """Validate a tool candidate. Raises InvalidToolError with specific message on failure."""
    if not hasattr(module, tool_name):
        raise InvalidToolError(
            f"Tool {tool_name!r}: module has no attribute named {tool_name!r}"
        )

    candidate = getattr(module, tool_name)

    if not callable(candidate):
        raise InvalidToolError(f"Tool {tool_name!r}: {tool_name!r} is not callable")

    if inspect.isclass(candidate):
        callable_to_check: Callable[..., object] = candidate.__call__
    else:
        callable_to_check = candidate

    if not callable_to_check.__doc__:
        raise InvalidToolError(
            f"Tool {tool_name!r}: missing docstring (required for tool description)"
        )

    try:
        hints = typing.get_type_hints(callable_to_check)
    except Exception as exc:
        raise InvalidToolError(
            f"Tool {tool_name!r}: failed to resolve type annotations: {exc}"
        ) from exc

    return_hint = hints.get("return")
    if return_hint is None:
        raise InvalidToolError(f"Tool {tool_name!r}: missing return type annotation")

    try:
        is_tool_result = issubclass(return_hint, ToolResult)
    except TypeError:
        is_tool_result = False
    if not is_tool_result:
        raise InvalidToolError(
            f"Tool {tool_name!r}: return type must be ToolResult, got {return_hint!r}"
        )

    sig = inspect.signature(callable_to_check)
    untyped = [name for name in sig.parameters if name != "self" and name not in hints]
    if untyped:
        raise InvalidToolError(
            f"Tool {tool_name!r}: untyped parameters: {', '.join(untyped)}"
        )
