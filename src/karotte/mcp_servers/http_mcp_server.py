import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from typing import Final, override

from karotte.log import configure_logging
from karotte.mcp_servers.mcp_server import McpServer
from karotte.schemas.http_mcp_server_config import HttpMcpServerConfig


class HttpMcpServer(McpServer):
    def __init__(self, config: HttpMcpServerConfig):
        super().__init__("karotte MCP Server")
        self.config: Final = config

        if config.profile_tool_calls:
            from karotte.mcp_servers.resource_profiling_middleware import (
                ResourceProfilingMiddleware,
            )

            self.server.add_middleware(ResourceProfilingMiddleware())

    @override
    def run(self):
        self.server.run(
            transport="streamable-http",
            host=self.config.host,
            port=self.config.port,
            show_banner=False,
            uvicorn_config={"access_log": False, "log_level": "warning"},
        )


@contextmanager
def run_server(config: HttpMcpServerConfig, suppress_output: bool = False):
    server = HttpMcpServer(config)

    # Create a subprocess to run the server
    cmd = [
        sys.executable,
        "-m",
        "karotte.mcp_servers.http_mcp_server",
        config.host,
        str(config.port),
        str(suppress_output),
        str(config.profile_tool_calls),
    ]

    stdout = subprocess.DEVNULL if suppress_output else None
    stderr = subprocess.DEVNULL if suppress_output else None

    process = subprocess.Popen(cmd, stdout=stdout, stderr=stderr)

    try:
        _wait_until_serving(process, config)
        yield server
    finally:
        process.terminate()
        process.wait(timeout=10)


def _wait_until_serving(
    process: subprocess.Popen[bytes], config: HttpMcpServerConfig, timeout: float = 30
) -> None:
    deadline = time.monotonic() + timeout
    while not _answers_http(config.client_url):
        if process.poll() is not None:
            raise RuntimeError(
                f"MCP server process exited unexpectedly with code {process.returncode}"
            )
        if time.monotonic() > deadline:
            raise TimeoutError(
                f"MCP server failed to start within {timeout} seconds. "
                + f"Check if port {config.port} is already in use or if there are startup issues."
            )
        time.sleep(0.02)


def _answers_http(url: str) -> bool:
    """Any HTTP response, even a rejection of the bare GET, means the app is serving."""
    try:
        urllib.request.urlopen(url, timeout=1.0).close()
    except urllib.error.HTTPError:
        return True
    except OSError:
        return False
    return True


def _subprocess_entrypoint():
    """Entrypoint for subprocess to run the MCP server."""
    import sys

    from karotte.container import is_containerized
    from karotte.run_helpers import sanitize_paths_and_reexec

    configure_logging()
    if is_containerized():
        sanitize_paths_and_reexec()

    # Read config from command line arguments
    host = sys.argv[1]
    port = int(sys.argv[2])
    suppress_output = sys.argv[3] == "True"
    profile_tool_calls = sys.argv[4] == "True" if len(sys.argv) > 4 else False

    config = HttpMcpServerConfig(
        host=host, port=port, profile_tool_calls=profile_tool_calls
    )
    server = HttpMcpServer(config)

    if suppress_output:
        _run_server_silently(server)
    else:
        server.run()


def _run_server_silently(server: HttpMcpServer):
    """Run an MCP server with suppressed output."""
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            server.run()
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


if __name__ == "__main__":
    _subprocess_entrypoint()
