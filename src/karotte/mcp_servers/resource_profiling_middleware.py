"""Middleware for profiling resource usage during tool execution."""

from typing import override

import mcp.types as mt
from fastmcp.server.middleware import CallNext, Middleware, MiddlewareContext
from fastmcp.tools.tool import ToolResult
from karotte.mcp_servers.resource_sampler import ResourceSampler


class ResourceProfilingMiddleware(Middleware):
    """Middleware that profiles CPU and memory usage for all tool calls.

    When enabled, this middleware wraps every tool execution with resource
    sampling, collecting CPU and memory metrics throughout the tool's execution.
    The metrics are injected into the tool result's structured_content.
    """

    @override
    async def on_call_tool(
        self,
        context: MiddlewareContext[mt.CallToolRequestParams],
        call_next: CallNext[mt.CallToolRequestParams, ToolResult],
    ) -> ToolResult:
        """Profile resource usage during tool execution."""
        # System-wide sampling (pid=None) captures resources from child processes
        async with ResourceSampler() as sampler:
            result = await call_next(context)

        # Always inject metrics - create structured_content if needed
        if result.structured_content is None:
            result.structured_content = {}

        result.structured_content["_karotte_resource_metrics"] = (
            sampler.metrics.model_dump()
        )

        return result
