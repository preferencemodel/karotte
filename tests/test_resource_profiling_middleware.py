from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from karotte.mcp_servers.resource_profiling_middleware import (
    ResourceProfilingMiddleware,
)


@dataclass
class MockToolResult:
    """Mock tool result with structured_content matching ToolResult's type."""

    structured_content: dict[str, Any] | None = field(default_factory=dict)


class TestResourceProfilingMiddleware:
    @pytest.mark.asyncio
    async def test_injects_metrics_into_structured_content(self):
        """Test that middleware injects _karotte_resource_metrics into result."""
        middleware = ResourceProfilingMiddleware()

        mock_result = MockToolResult(structured_content={"stdout": "hello"})
        mock_call_next = AsyncMock(return_value=mock_result)
        mock_context = MagicMock()

        result = await middleware.on_call_tool(mock_context, mock_call_next)

        assert result.structured_content is not None
        assert "_karotte_resource_metrics" in result.structured_content
        metrics = result.structured_content["_karotte_resource_metrics"]
        assert "samples" in metrics
        assert "peak_cpu_percent" in metrics
        assert "avg_cpu_percent" in metrics
        assert "peak_memory_mb" in metrics
        assert "avg_memory_mb" in metrics

    @pytest.mark.asyncio
    async def test_preserves_existing_structured_content(self):
        """Test that existing structured_content fields are preserved."""
        middleware = ResourceProfilingMiddleware()

        mock_result = MockToolResult(
            structured_content={"stdout": "hello", "stderr": ""}
        )
        mock_call_next = AsyncMock(return_value=mock_result)
        mock_context = MagicMock()

        result = await middleware.on_call_tool(mock_context, mock_call_next)

        assert result.structured_content is not None
        assert result.structured_content["stdout"] == "hello"
        assert result.structured_content["stderr"] == ""
        assert "_karotte_resource_metrics" in result.structured_content

    @pytest.mark.asyncio
    async def test_handles_none_structured_content(self):
        """Test that middleware creates dict and injects when structured_content is None."""
        middleware = ResourceProfilingMiddleware()

        mock_result = MockToolResult(structured_content=None)
        mock_call_next = AsyncMock(return_value=mock_result)
        mock_context = MagicMock()

        result = await middleware.on_call_tool(mock_context, mock_call_next)

        # Should create dict and inject metrics
        assert result.structured_content is not None
        assert isinstance(result.structured_content, dict)
        assert "_karotte_resource_metrics" in result.structured_content

    @pytest.mark.asyncio
    async def test_calls_next_middleware(self):
        """Test that middleware calls the next handler in chain."""
        middleware = ResourceProfilingMiddleware()

        mock_result = MockToolResult(structured_content={})
        mock_call_next = AsyncMock(return_value=mock_result)
        mock_context = MagicMock()

        _ = await middleware.on_call_tool(mock_context, mock_call_next)

        mock_call_next.assert_called_once_with(mock_context)
