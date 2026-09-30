from fastmcp.tools.tool import ToolResult


def empty_result() -> ToolResult:
    """Returns a result with no content blocks."""
    return ToolResult(content=[])
