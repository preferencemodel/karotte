from fastmcp.tools.tool import ToolResult


def echo(text: str) -> ToolResult:
    """Echoes `text` back to the caller."""
    return ToolResult(structured_content={"echo": text})
