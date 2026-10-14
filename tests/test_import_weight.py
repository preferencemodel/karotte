"""Importing the CLI or karotte's schemas must not load an LLM client, an MCP server or the Modal SDK."""

import subprocess
import sys

import pytest

HEAVY_MODULES = ["litellm", "anthropic", "openai", "mcp.server", "modal"]


@pytest.mark.parametrize(
    "module",
    [
        "karotte",
        "karotte.cli",
        "karotte.schemas.transcript",
        "karotte.schemas.evaluation_run_config",
        "karotte.model_catalog",
        "karotte.transcript_markdown",
    ],
)
def test_importing_karotte_loads_no_llm_client(module: str):
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            f"import sys, {module}; "
            + f"print(' '.join(m for m in {HEAVY_MODULES!r} if m in sys.modules))",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == ""
