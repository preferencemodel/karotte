from fastmcp.tools.tool import ToolResult
from mcp.types import ImageContent

# 1x1 pixel PNG, fixed so tool output is byte-identical across runs.
_PNG_1PX_BASE64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mMIDLL5DwADtgHfYCywCgAAAABJRU5ErkJggg=="


def get_image() -> ToolResult:
    """Returns a fixed 1x1 pixel PNG image."""
    return ToolResult(
        content=[ImageContent(type="image", data=_PNG_1PX_BASE64, mimeType="image/png")]
    )
