import inspect
from typing import Final

from fastmcp import FastMCP
from karotte.mcp_servers.discover_tools import discover_tools
from loguru import logger


class McpServer:
    def __init__(self, name: str):
        self.server: Final = FastMCP(name)
        self.registered_tools: set[str] = set()

        # Cannot directly decorate the method.
        # See https://gofastmcp.com/patterns/decorating-methods
        self.server.tool()(self.register_tools)

    def register_tools(
        self,
        tools: list[str] | None = None,
        name_overrides: dict[str, str] | None = None,
    ) -> None:
        """Registers `tools` with `server`.

        Args:
            tools: A list of tool names to register. If None, all available tools will be registered.
        """
        self._register_tools(tools, name_overrides)
        try:
            self.server.local_provider.remove_tool("register_tools")
        except KeyError:
            logger.info("register_tools tool already removed from MCP server")

    def run(self) -> None:
        raise NotImplementedError("Sub-classes must implement this")

    def _register_tools(
        self, tools: list[str] | None, name_overrides: dict[str, str] | None = None
    ):
        name_overrides = name_overrides or {}

        for tool_name, module in discover_tools(tools).items():
            candidate = getattr(module, tool_name)

            if inspect.isclass(candidate):
                # Only instance methods can be registered as tools
                # https://gofastmcp.com/patterns/decorating-methods
                candidate = candidate().__call__

            self.server.tool(candidate, name=name_overrides.get(tool_name, tool_name))
            self.registered_tools.add(tool_name)
