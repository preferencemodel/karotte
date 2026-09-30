from pydantic import BaseModel


class HttpMcpServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8080
    profile_tool_calls: bool = False
    """Whether to collect CPU/memory metrics during tool execution."""

    @property
    def client_url(self) -> str:
        """URL a co-located client uses to reach the server.

        The server may bind an unspecified address (0.0.0.0/::), which the
        server's Host-header guard never trusts. A local client must connect
        over loopback so its Host header is accepted."""
        host = "127.0.0.1" if self.host in ("0.0.0.0", "::", "") else self.host
        return f"http://{host}:{self.port}/mcp"
