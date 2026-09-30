import importlib
import sys
from pathlib import Path
from types import ModuleType
from typing import Annotated

import pytest
from fastmcp.tools.tool import ToolResult

from karotte.mcp_servers.discover_tools import (
    InvalidToolError,
    _filter_candidates,  # pyright: ignore[reportPrivateUsage]
    _validate_tool,  # pyright: ignore[reportPrivateUsage]
    discover_tools,
)


@pytest.mark.asyncio
async def test_tool_discovery():
    tools = discover_tools()

    assert "bash" in tools
    assert "replace_in_file" in tools
    assert "view_lines_in_file" in tools


class TestDiscoverToolsWithExplicitNames:
    """Tests for discover_tools when tool_names is provided."""

    def test_returns_only_requested_tools(self):
        """When tool_names is provided, only those tools are returned."""
        tools = discover_tools(tool_names=["bash", "replace_in_file"])

        assert list(tools.keys()) == ["bash", "replace_in_file"]

    def test_raises_for_nonexistent_tool(self):
        """When a requested tool doesn't exist, raises InvalidToolError."""
        with pytest.raises(InvalidToolError, match="Tool 'nonexistent_tool' not found"):
            discover_tools(tool_names=["nonexistent_tool"])

    def test_preserves_order(self):
        """Returned tools preserve the order of tool_names."""
        tools = discover_tools(tool_names=["replace_in_file", "bash"])

        assert list(tools.keys()) == ["replace_in_file", "bash"]


class TestFilterCandidates:
    """Tests for _filter_candidates function."""

    def test_filters_to_requested_tools(self):
        """Should return only the requested tools."""
        candidates: dict[str, object] = {"a": "mod_a", "b": "mod_b", "c": "mod_c"}
        result = _filter_candidates(candidates, ["a", "c"])

        assert result == {"a": "mod_a", "c": "mod_c"}

    def test_preserves_order_of_tool_names(self):
        """Should preserve the order of tool_names in the result."""
        candidates: dict[str, object] = {"a": "mod_a", "b": "mod_b", "c": "mod_c"}
        result = _filter_candidates(candidates, ["c", "a"])

        assert list(result.keys()) == ["c", "a"]

    def test_raises_for_missing_tool(self):
        """Should raise InvalidToolError if a requested tool is missing."""
        candidates: dict[str, object] = {"a": "mod_a", "b": "mod_b"}

        with pytest.raises(InvalidToolError, match="Tool 'missing' not found"):
            _filter_candidates(candidates, ["a", "missing"])

    def test_empty_tool_names_returns_empty(self):
        """Should return empty dict when tool_names is empty."""
        candidates: dict[str, object] = {"a": "mod_a", "b": "mod_b"}
        result = _filter_candidates(candidates, [])

        assert result == {}


# --- Tests for _validate_tool ---


def _create_module_with_attr(name: str, attr: object) -> ModuleType:
    """Helper to create a module with a single attribute."""
    module = ModuleType(name)
    setattr(module, name, attr)
    return module


class TestValidateTool:
    """Tests for _validate_tool function."""

    def test_missing_attribute_raises(self):
        """Module without matching attribute should raise."""
        module = ModuleType("my_tool")
        with pytest.raises(
            InvalidToolError, match="module has no attribute named 'my_tool'"
        ):
            _validate_tool(module, "my_tool")

    def test_non_callable_raises(self):
        """Non-callable attribute should raise."""
        module = _create_module_with_attr("my_tool", "not a callable")
        with pytest.raises(InvalidToolError, match="'my_tool' is not callable"):
            _validate_tool(module, "my_tool")

    def test_function_without_docstring_raises(self):
        """Function without docstring should raise."""

        def my_tool() -> ToolResult:
            return ToolResult(content=[])

        module = _create_module_with_attr("my_tool", my_tool)
        with pytest.raises(InvalidToolError, match="missing docstring"):
            _validate_tool(module, "my_tool")

    def test_class_without_docstring_raises(self):
        """Class with __call__ without docstring should raise."""

        class my_tool:
            def __call__(self) -> ToolResult:
                return ToolResult(content=[])

        module = _create_module_with_attr("my_tool", my_tool)
        with pytest.raises(InvalidToolError, match="missing docstring"):
            _validate_tool(module, "my_tool")

    def test_function_without_return_annotation_raises(self):
        """Function without return annotation should raise."""

        def my_tool():
            """A tool."""

        module = _create_module_with_attr("my_tool", my_tool)
        with pytest.raises(InvalidToolError, match="missing return type annotation"):
            _validate_tool(module, "my_tool")

    def test_function_with_wrong_return_type_raises(self):
        """Function returning non-ToolResult should raise."""

        def my_tool() -> str:
            """A tool."""
            return "not a tool result"

        module = _create_module_with_attr("my_tool", my_tool)
        with pytest.raises(InvalidToolError, match="return type must be ToolResult"):
            _validate_tool(module, "my_tool")

    def test_valid_function_tool_passes(self):
        """Valid function tool with ToolResult return type should pass."""

        def my_tool() -> ToolResult:
            """A tool."""
            return ToolResult(content=[])

        module = _create_module_with_attr("my_tool", my_tool)
        _validate_tool(module, "my_tool")  # Should not raise

    def test_valid_function_tool_with_typed_params_passes(self):
        """Function with typed parameters should pass."""

        def my_tool(x: int, y: str) -> ToolResult:  # pyright: ignore[reportUnusedParameter]
            """A tool."""
            return ToolResult(content=[])

        module = _create_module_with_attr("my_tool", my_tool)
        _validate_tool(module, "my_tool")  # Should not raise

    def test_function_with_annotated_params_passes(self):
        """Function with Annotated parameters should pass."""

        def my_tool(
            x: Annotated[int, "an integer"],  # pyright: ignore[reportUnusedParameter]
            y: Annotated[str, "a string"],  # pyright: ignore[reportUnusedParameter]
        ) -> ToolResult:
            """A tool."""
            return ToolResult(content=[])

        module = _create_module_with_attr("my_tool", my_tool)
        _validate_tool(module, "my_tool")  # Should not raise

    def test_function_with_default_values_passes(self):
        """Function with default values and type annotations should pass."""

        def my_tool(x: int = 10, y: str = "default") -> ToolResult:  # pyright: ignore[reportUnusedParameter]
            """A tool."""
            return ToolResult(content=[])

        module = _create_module_with_attr("my_tool", my_tool)
        _validate_tool(module, "my_tool")  # Should not raise

    def test_valid_class_tool_passes(self):
        """Valid class tool with __call__ returning ToolResult should pass."""

        class my_tool:
            def __call__(self) -> ToolResult:
                """A tool."""
                return ToolResult(content=[])

        module = _create_module_with_attr("my_tool", my_tool)
        _validate_tool(module, "my_tool")  # Should not raise

    def test_class_tool_with_typed_params_passes(self):
        """Class tool with typed __call__ parameters should pass."""

        class my_tool:
            def __call__(self, x: int, y: str) -> ToolResult:
                """A tool."""
                return ToolResult(content=[])

        module = _create_module_with_attr("my_tool", my_tool)
        _validate_tool(module, "my_tool")  # Should not raise

    def test_class_without_call_return_annotation_raises(self):
        """Class with __call__ missing return annotation should raise."""

        class my_tool:
            def __call__(self):
                """A tool."""

        module = _create_module_with_attr("my_tool", my_tool)
        with pytest.raises(InvalidToolError, match="missing return type annotation"):
            _validate_tool(module, "my_tool")

    def test_class_with_wrong_return_type_raises(self):
        """Class with __call__ returning wrong type should raise."""

        class my_tool:
            def __call__(self) -> str:
                """A tool."""
                return "not a tool result"

        module = _create_module_with_attr("my_tool", my_tool)
        with pytest.raises(InvalidToolError, match="return type must be ToolResult"):
            _validate_tool(module, "my_tool")


class TestEnvironmentToolsDiscovery:
    """Tests for discovering tools from environment.tools package."""

    @pytest.fixture(autouse=True)
    def _clean_environment_modules(self):
        yield
        modules_to_remove = [k for k in sys.modules if k.startswith("environment")]
        for mod in modules_to_remove:
            del sys.modules[mod]

    def test_discovers_tools_from_environment_package(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        """Should discover tools from environment.tools when it exists."""
        # Create a mock environment.tools package
        env_tools_dir = tmp_path / "environment" / "tools"
        env_tools_dir.mkdir(parents=True)

        # Create __init__.py files
        (tmp_path / "environment" / "__init__.py").write_text("")
        (env_tools_dir / "__init__.py").write_text("")

        # Create a valid tool module
        tool_code = '''
from fastmcp.tools.tool import ToolResult

def custom_tool(param: str) -> ToolResult:
    """A custom tool from environment.tools."""
    return ToolResult(content=[{"type": "text", "text": f"Custom: {param}"}])
'''
        (env_tools_dir / "custom_tool.py").write_text(tool_code)

        # Add tmp_path to sys.path so environment.tools can be imported
        import sys

        monkeypatch.syspath_prepend(str(tmp_path))

        # Clear any cached imports
        if "environment.tools" in sys.modules:
            del sys.modules["environment.tools"]
        if "environment" in sys.modules:
            del sys.modules["environment"]

        # Discover tools - should include both karotte.tools and environment.tools
        tools = discover_tools()

        assert "custom_tool" in tools
        assert "bash" in tools  # From karotte.tools

    def test_imports_environment_tools_with_correct_prefix(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        """Should import environment.tools modules with environment.tools prefix."""
        from karotte.mcp_servers.discover_tools import (
            _collect_all_candidates,  # pyright: ignore[reportPrivateUsage]
        )

        # Create a mock environment.tools package
        env_tools_dir = tmp_path / "environment" / "tools"
        env_tools_dir.mkdir(parents=True)

        (tmp_path / "environment" / "__init__.py").write_text("")
        (env_tools_dir / "__init__.py").write_text("")

        tool_code = '''
from fastmcp.tools.tool import ToolResult

def env_specific_tool(x: int) -> ToolResult:
    """Environment specific tool."""
    return ToolResult(content=[{"type": "text", "text": str(x)}])
'''
        (env_tools_dir / "env_specific_tool.py").write_text(tool_code)

        import sys

        monkeypatch.syspath_prepend(str(tmp_path))

        # Clear cached imports
        if "environment.tools" in sys.modules:
            del sys.modules["environment.tools"]
        if "environment" in sys.modules:
            del sys.modules["environment"]

        candidates = _collect_all_candidates()

        # Verify the module was imported
        assert "env_specific_tool" in candidates

        # Verify it's the correct module (imported with environment.tools prefix)
        module = candidates["env_specific_tool"]
        assert module.__name__ == "environment.tools.env_specific_tool"  # pyright: ignore[reportAttributeAccessIssue]

    def test_handles_missing_environment_tools_gracefully(self):
        """Should work normally when environment.tools doesn't exist."""
        import sys

        # Ensure environment.tools is not importable
        if "environment.tools" in sys.modules:
            del sys.modules["environment.tools"]
        if "environment" in sys.modules:
            del sys.modules["environment"]

        # Should not raise, just discover karotte.tools
        tools = discover_tools()

        # Should still find karotte.tools
        assert "bash" in tools
        assert "replace_in_file" in tools

    def test_environment_tool_overrides_karotte_tool(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        """When same tool name exists in both, environment.tools takes precedence."""
        # Create environment.tools with a tool that has the same name as a karotte tool
        env_tools_dir = tmp_path / "environment" / "tools"
        env_tools_dir.mkdir(parents=True)

        (tmp_path / "environment" / "__init__.py").write_text("")
        (env_tools_dir / "__init__.py").write_text("")

        # Create a tool with the same name as an existing karotte tool
        tool_code = '''
from fastmcp.tools.tool import ToolResult

def bash(command: str) -> ToolResult:
    """Custom bash tool from environment."""
    return ToolResult(content=[{"type": "text", "text": "Custom bash"}])
'''
        (env_tools_dir / "bash.py").write_text(tool_code)

        import sys

        monkeypatch.syspath_prepend(str(tmp_path))

        if "environment.tools" in sys.modules:
            del sys.modules["environment.tools"]
        if "environment" in sys.modules:
            del sys.modules["environment"]

        from karotte.mcp_servers.discover_tools import (
            _collect_all_candidates,  # pyright: ignore[reportPrivateUsage]
        )

        candidates = _collect_all_candidates()

        # The environment.tools version should override
        bash_module = candidates["bash"]
        assert bash_module.__name__ == "environment.tools.bash"  # pyright: ignore[reportAttributeAccessIssue]


class TestUntypedParameterDetection:
    """Tests for untyped parameter detection and error raising."""

    def test_function_with_untyped_param_raises(self):
        """Function with untyped parameter should raise InvalidToolError."""

        def my_tool(x) -> ToolResult:  # pyright: ignore[reportMissingParameterType, reportUnusedParameter]
            """A tool."""
            return ToolResult(content=[])

        module = _create_module_with_attr("my_tool", my_tool)

        with pytest.raises(InvalidToolError, match="untyped parameters: x"):
            _validate_tool(module, "my_tool")

    def test_function_with_multiple_untyped_params_raises(self):
        """Function with multiple untyped parameters should list all in error."""

        def my_tool(a, b, c) -> ToolResult:  # pyright: ignore[reportMissingParameterType, reportUnusedParameter]
            """A tool."""
            return ToolResult(content=[])

        module = _create_module_with_attr("my_tool", my_tool)

        with pytest.raises(InvalidToolError, match="untyped parameters: a, b, c"):
            _validate_tool(module, "my_tool")

    def test_function_with_mixed_typed_untyped_params_raises(self):
        """Function with mix of typed and untyped parameters should raise."""

        def my_tool(x: int, y, z: str) -> ToolResult:  # pyright: ignore[reportUnusedParameter, reportMissingParameterType]
            """A tool."""
            return ToolResult(content=[])

        module = _create_module_with_attr("my_tool", my_tool)

        with pytest.raises(InvalidToolError, match="untyped parameters: y"):
            _validate_tool(module, "my_tool")

    def test_class_with_untyped_call_param_raises(self):
        """Class with untyped __call__ parameter should raise InvalidToolError."""

        class my_tool:
            def __call__(self, x) -> ToolResult:  # pyright: ignore[reportMissingParameterType]
                """A tool."""
                return ToolResult(content=[])

        module = _create_module_with_attr("my_tool", my_tool)

        with pytest.raises(InvalidToolError, match="untyped parameters: x"):
            _validate_tool(module, "my_tool")

    def test_class_self_param_is_ignored(self):
        """The 'self' parameter should not require type annotation."""

        class my_tool:
            def __call__(self) -> ToolResult:
                """A tool."""
                return ToolResult(content=[])

        module = _create_module_with_attr("my_tool", my_tool)
        _validate_tool(module, "my_tool")  # Should not raise

    def test_error_message_contains_tool_name(self):
        """Error message should include the tool name for debugging."""

        def special_tool_name(x) -> ToolResult:  # pyright: ignore[reportMissingParameterType, reportUnusedParameter]
            """A tool."""
            return ToolResult(content=[])

        module = _create_module_with_attr("special_tool_name", special_tool_name)

        with pytest.raises(InvalidToolError, match="Tool 'special_tool_name'"):
            _validate_tool(module, "special_tool_name")


class TestGenericReturnTypeError:
    """Non-class return annotations (e.g. generics) should produce a clear error."""

    def test_optional_return_type_gives_invalid_tool_error(self):
        """Using Optional[ToolResult] as return type should raise InvalidToolError, not TypeError."""

        def bad_tool() -> ToolResult | None:
            """A tool with a union return type."""
            return None

        module = _create_module_with_attr("bad_tool", bad_tool)

        with pytest.raises(InvalidToolError, match="return type must be ToolResult"):
            _validate_tool(module, "bad_tool")


class TestFutureAnnotationsCompatibility:
    """Tools using `from __future__ import annotations` must pass validation."""

    def test_function_tool_with_future_annotations(self, tmp_path: Path):
        """A function tool in a module with PEP 563 deferred annotations should validate."""
        tool_code = '''\
from __future__ import annotations

from fastmcp.tools.tool import ToolResult

def future_tool(x: int, y: str) -> ToolResult:
    """A tool that uses future annotations."""
    return ToolResult(content=[])
'''
        (tmp_path / "future_tool.py").write_text(tool_code)
        sys.path.insert(0, str(tmp_path))
        try:
            module = importlib.import_module("future_tool")
            _validate_tool(module, "future_tool")  # Should not raise
        finally:
            sys.path.remove(str(tmp_path))
            sys.modules.pop("future_tool", None)

    def test_class_tool_with_future_annotations(self, tmp_path: Path):
        """A class tool in a module with PEP 563 deferred annotations should validate."""
        tool_code = '''\
from __future__ import annotations

from fastmcp.tools.tool import ToolResult

class class_tool:
    def __call__(self, x: int) -> ToolResult:
        """A class-based tool that uses future annotations."""
        return ToolResult(content=[])
'''
        (tmp_path / "class_tool.py").write_text(tool_code)
        sys.path.insert(0, str(tmp_path))
        try:
            module = importlib.import_module("class_tool")
            _validate_tool(module, "class_tool")  # Should not raise
        finally:
            sys.path.remove(str(tmp_path))
            sys.modules.pop("class_tool", None)
