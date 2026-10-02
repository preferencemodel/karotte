import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastmcp import Client
from mcp.types import TextContent

from karotte.mcp_servers.http_mcp_server import HttpMcpServer, run_server
from karotte.mcp_servers.mcp_server import McpServer
from karotte.schemas.http_mcp_server_config import HttpMcpServerConfig


def test_client_url_maps_unspecified_bind_host_to_loopback():
    """The server binds an unspecified address, which its Host-header guard
    never trusts; a co-located client must connect over loopback."""
    assert (
        HttpMcpServerConfig(host="0.0.0.0", port=8080).client_url
        == "http://127.0.0.1:8080/mcp"
    )
    assert (
        HttpMcpServerConfig(host="::", port=9000).client_url
        == "http://127.0.0.1:9000/mcp"
    )


def test_client_url_preserves_explicit_host():
    """An explicitly configured host is added to the guard's allowed hosts, so
    the client keeps using it verbatim."""
    assert (
        HttpMcpServerConfig(host="mcp.internal", port=8080).client_url
        == "http://mcp.internal:8080/mcp"
    )


@pytest.mark.asyncio
async def test_mcp_server_register_function_tool(tmp_path: Path):
    mcp_server = McpServer("")
    mcp_server.register_tools(["replace_in_file"])

    assert mcp_server.registered_tools == {"replace_in_file"}

    file = tmp_path / "test.txt"
    file.write_text("Hello, world!")

    async with Client(mcp_server.server) as client:
        # Valid tool call
        result = await client.call_tool(
            "replace_in_file",
            {"file_path": str(file), "old": "world", "new": "universe"},
        )
        assert result.data["result"] == "Replacement successful"
        assert result.is_error is False

        # Invalid tool call
        result = await client.call_tool_mcp(
            "replace_in_file", {"file": 12, "old": "world", "new": "universe"}
        )
        assert result.isError is True


@pytest.mark.asyncio
async def test_register_class_tool():
    mcp_server = HttpMcpServer(config=HttpMcpServerConfig())
    mcp_server.register_tools(["bash"])

    async with Client(mcp_server.server) as client:
        result = await client.call_tool("bash", arguments={"command": "Hello world"})

        assert isinstance(result.content[0], TextContent)
        assert json.loads(result.content[0].text)["stdout"] == ""


@pytest.mark.asyncio
async def test_register_tools_with_name_override():
    mcp_server = McpServer("")
    mcp_server.register_tools(["bash"], name_overrides={"bash": "shell"})

    async with Client(mcp_server.server) as client:
        tools = await client.list_tools()
        tool_names = {tool.name for tool in tools}

        assert "shell" in tool_names
        assert "bash" not in tool_names


@pytest.mark.asyncio
async def test_register_custom_environment_tool(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """Registering a tool from environment.tools should not fail."""
    env_tools_dir = tmp_path / "environment" / "tools"
    env_tools_dir.mkdir(parents=True)
    (tmp_path / "environment" / "__init__.py").write_text("")
    (env_tools_dir / "__init__.py").write_text("")

    tool_code = '''\
from fastmcp.tools.tool import ToolResult

def custom_tool(param: str) -> ToolResult:
    """A custom tool from environment.tools."""
    return ToolResult(content=[{"type": "text", "text": f"Custom: {param}"}])
'''
    (env_tools_dir / "custom_tool.py").write_text(tool_code)

    monkeypatch.syspath_prepend(str(tmp_path))

    # Clear cached imports so environment.tools is discovered fresh
    for mod_name in [k for k in sys.modules if k.startswith("environment")]:
        del sys.modules[mod_name]

    try:
        mcp_server = McpServer("")
        mcp_server.register_tools(["custom_tool"])

        assert "custom_tool" in mcp_server.registered_tools

        async with Client(mcp_server.server) as client:
            result = await client.call_tool("custom_tool", arguments={"param": "hello"})
            assert isinstance(result.content[0], TextContent)
            assert "Custom: hello" in result.content[0].text
    finally:
        for mod_name in [k for k in sys.modules if k.startswith("environment")]:
            del sys.modules[mod_name]


@pytest.mark.asyncio
async def test_register_tools_can_only_be_called_once():
    mcp_server = McpServer("")

    async with Client(mcp_server.server) as client:
        tools_before = await client.list_tools()
        tool_names_before = {tool.name for tool in tools_before}
        assert "register_tools" in tool_names_before

        await client.call_tool("register_tools", arguments={"tools": []})

        tools_after = await client.list_tools()
        tool_names_after = {tool.name for tool in tools_after}
        assert "register_tools" not in tool_names_after


@pytest.mark.asyncio
async def test_register_tools_twice_does_not_raise():
    mcp_server = McpServer("")
    mcp_server.register_tools(["bash"])
    mcp_server.register_tools(["bash"])


def test_subprocess_command_line_carries_no_task_id():
    """/proc/[pid]/cmdline is world-readable, so the student can read the MCP
    server's argv; a task id there can name the intended solution."""
    config = HttpMcpServerConfig(host="127.0.0.1", port=8099)
    process = MagicMock(poll=MagicMock(return_value=None))

    with (
        patch(
            "karotte.mcp_servers.http_mcp_server.subprocess.Popen", return_value=process
        ) as mock_popen,
        patch("karotte.mcp_servers.http_mcp_server._answers_http", return_value=True),
    ):
        with run_server(config):
            pass

    cmd = mock_popen.call_args[0][0]
    assert "example-task" not in cmd
    assert all("example-task" not in arg for arg in cmd)


def test_run_server_yields_as_soon_as_the_server_answers_http():
    import http.server
    import threading
    import time

    class _Reject(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_error(406)

        def log_message(self, format: str, *args: object) -> None:
            pass

    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Reject)
    threading.Thread(
        target=httpd.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
    ).start()
    config = HttpMcpServerConfig(host="127.0.0.1", port=httpd.server_address[1])
    process = MagicMock(poll=MagicMock(return_value=None))

    try:
        start = time.perf_counter()
        with patch(
            "karotte.mcp_servers.http_mcp_server.subprocess.Popen", return_value=process
        ):
            with run_server(config):
                elapsed = time.perf_counter() - start
    finally:
        httpd.shutdown()

    assert elapsed < 0.3


def test_http_server_runs_without_banner_or_info_logs():
    server = HttpMcpServer(HttpMcpServerConfig(host="127.0.0.1", port=8099))

    with patch.object(server.server, "run") as run:
        server.run()

    kwargs = run.call_args.kwargs
    assert kwargs["show_banner"] is False
    assert kwargs["uvicorn_config"] == {"access_log": False, "log_level": "warning"}


def test_subprocess_entrypoint_configures_logging(monkeypatch: pytest.MonkeyPatch):
    import karotte.mcp_servers.http_mcp_server as http_mcp_server

    calls: list[str] = []
    monkeypatch.setattr(
        http_mcp_server, "configure_logging", lambda: calls.append("log")
    )

    def fake_run(_self: HttpMcpServer) -> None:
        calls.append("run")

    monkeypatch.setattr(HttpMcpServer, "run", fake_run)
    monkeypatch.setattr(sys, "argv", ["x", "127.0.0.1", "8099", "False", "False"])

    http_mcp_server._subprocess_entrypoint()  # pyright: ignore[reportPrivateUsage]

    assert calls == ["log", "run"]
