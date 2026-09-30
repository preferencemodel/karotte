"""Import ``mcp.types`` without triggering the heavy ``mcp.__init__``.

``mcp.__init__`` eagerly imports client sessions, server sessions, uvicorn,
starlette, etc., adding ~800 ms of import time.  All we need from ``mcp`` in
the schemas package is the lightweight ``mcp.types`` module.  This helper
inserts a lightweight placeholder package so that ``mcp.types`` can be loaded
in isolation.

If anything later needs the *full* ``mcp`` package (e.g. ``fastmcp``), the
placeholder transparently loads the real ``mcp.__init__`` on first access of
any attribute other than ``types``.
"""

import importlib
import importlib.util
import sys
import types
from typing import Any


class _LazyMcpPlaceholder(types.ModuleType):
    """Stands in for the ``mcp`` package until the real one is needed.

    Submodule imports (``from mcp.client.session import …``) work because
    ``__path__`` is set.  Attribute access like ``mcp.ClientSession`` triggers
    loading the real ``mcp/__init__.py`` on demand.
    """

    _real: types.ModuleType | None = None

    def _load_real(self) -> types.ModuleType:
        if self._real is not None:
            return self._real

        # Temporarily remove ourselves so the real __init__ can run.
        sys.modules.pop("mcp", None)
        real = importlib.import_module("mcp")

        # CPython skips the parent-attribute fixup for submodules already in
        # sys.modules.  Patch up any that were loaded before the real init ran.
        for key, mod in sys.modules.items():
            if key.startswith("mcp.") and not hasattr(real, key.split(".")[-1]):
                setattr(real, key.split(".")[-1], mod)

        self._real = real
        return real

    def __getattr__(self, name: str) -> Any:
        return getattr(self._load_real(), name)


def import_mcp_types() -> types.ModuleType:
    """Return the ``mcp.types`` module, importing only what is necessary."""
    if "mcp.types" in sys.modules:
        return sys.modules["mcp.types"]

    had_mcp = "mcp" in sys.modules

    if not had_mcp:
        spec = importlib.util.find_spec("mcp")
        assert spec is not None and spec.submodule_search_locations is not None
        placeholder = _LazyMcpPlaceholder("mcp")
        placeholder.__path__ = list(spec.submodule_search_locations)  # type: ignore[assignment]
        placeholder.__package__ = "mcp"
        placeholder.__spec__ = spec
        sys.modules["mcp"] = placeholder

    return importlib.import_module("mcp.types")


def get_mcp_type(name: str) -> Any:
    """Return a single type from ``mcp.types`` by *name*."""
    return getattr(import_mcp_types(), name)
