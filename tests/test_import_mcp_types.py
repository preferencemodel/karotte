import subprocess
import sys

_PLACEHOLDER_CARRIES_SPEC = """
import sys

assert "mcp" not in sys.modules
from karotte.schemas._import_mcp_types import import_mcp_types

import_mcp_types()
placeholder = sys.modules["mcp"]
# fastmcp >= 4 reads mcp.__spec__ at import time and raises on None, and
# since the attribute exists on a bare ModuleType, __getattr__ never forwards
# it to the real module — the placeholder must carry the real spec itself.
assert placeholder.__spec__ is not None, "placeholder mcp module lost its __spec__"
assert placeholder.__spec__.submodule_search_locations
"""


def test_mcp_placeholder_carries_the_real_spec():
    """The lazy ``mcp`` placeholder must look like the real package to
    importers that inspect ``__spec__`` (fastmcp 4 crashed on None)."""
    # A subprocess gives an interpreter where ``mcp`` isn't imported yet;
    # in-process, conftest/other tests may already have loaded the real thing.
    subprocess.run(
        (sys.executable, "-c", _PLACEHOLDER_CARRIES_SPEC),
        check=True,
    )
