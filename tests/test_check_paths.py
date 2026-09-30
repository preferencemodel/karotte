"""Build-time layer: the image must never ship a loader path with an empty or relative component (which resolves the cwd).

`check_paths()` is the build-time gate; the Containerfile tests assert the default template can't ship such a value.
"""

from pathlib import Path

import pytest
from jinja2 import Environment, FileSystemLoader

from karotte.check_paths import UnsafeLoaderPath, check_paths

TEMPLATES = Path(__file__).resolve().parent.parent / "src/karotte/templates"
CONTAINERFILE = TEMPLATES / "default/Containerfile"

_CHECKED_VARS = ("PATH", "LD_LIBRARY_PATH", "LD_PRELOAD", "LD_AUDIT")


class TestCheckPaths:
    def test_accepts_absolute_only(self):
        check_paths(
            {
                "PATH": "/usr/local/bin:/usr/bin",
                "LD_LIBRARY_PATH": "/usr/local/nvidia/lib64",
                "LD_PRELOAD": "",
                "LD_AUDIT": "",
            }
        )

    def test_accepts_missing_vars(self):
        check_paths({})

    def test_rejects_trailing_colon(self):
        with pytest.raises(UnsafeLoaderPath):
            check_paths({"LD_LIBRARY_PATH": "/usr/local/nvidia/lib64:"})

    def test_rejects_leading_colon(self):
        with pytest.raises(UnsafeLoaderPath):
            check_paths({"PATH": ":/usr/bin"})

    def test_rejects_doubled_colon(self):
        with pytest.raises(UnsafeLoaderPath):
            check_paths({"PATH": "/usr/local/bin::/usr/bin"})

    def test_rejects_relative_entry(self):
        with pytest.raises(UnsafeLoaderPath):
            check_paths({"LD_LIBRARY_PATH": "/usr/lib:lib64"})

    def test_rejects_dot_entry(self):
        with pytest.raises(UnsafeLoaderPath):
            check_paths({"PATH": "/usr/bin:."})

    @pytest.mark.parametrize("var", _CHECKED_VARS)
    def test_checks_every_loader_var(self, var: str):
        with pytest.raises(UnsafeLoaderPath):
            check_paths({var: "/good:"})

    def test_rejects_space_separated_relative_ld_preload(self):
        """glibc splits LD_PRELOAD on spaces as well as colons."""
        with pytest.raises(UnsafeLoaderPath):
            check_paths({"LD_PRELOAD": "/abs/a.so evil.so"})

    def test_accepts_space_separated_absolute_ld_preload(self):
        check_paths({"LD_PRELOAD": "/abs/a.so /abs/b.so"})

    def test_space_separation_only_applies_to_ld_preload(self):
        """Directory names may contain spaces; PATH-like vars split on colons only."""
        check_paths(
            {"PATH": "/opt/some dir/bin", "LD_LIBRARY_PATH": "/opt/some dir/lib"}
        )

    def test_error_names_the_offending_var(self):
        with pytest.raises(UnsafeLoaderPath, match="LD_PRELOAD"):
            check_paths({"LD_PRELOAD": "relative.so"})

    def test_defaults_to_process_environment(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("LD_LIBRARY_PATH", "/opt/lib:")
        with pytest.raises(UnsafeLoaderPath):
            check_paths()


class TestContainerfileShipsCleanLoaderPaths:
    """Templates add loader-path ENVs through `partials/Containerfile.jinja`; `karotte check` must run after them."""

    PARTIAL: str = "ARG LD_LIBRARY_PATH\nENV LD_LIBRARY_PATH=/opt/x/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}\n"

    @pytest.fixture(scope="class")
    def rendered(self, tmp_path_factory: pytest.TempPathFactory) -> str:
        extra = tmp_path_factory.mktemp("templates")
        (extra / "gpu" / "partials").mkdir(parents=True)
        (extra / "gpu" / "partials" / "Containerfile.jinja").write_text(self.PARTIAL)
        env = Environment(
            loader=FileSystemLoader([TEMPLATES, extra]), keep_trailing_newline=True
        )
        return env.get_template("default/Containerfile").render(
            templates=["default", "gpu"]
        )

    def test_default_names_no_gpu_driver_paths(self):
        assert "nvidia" not in CONTAINERFILE.read_text()

    def test_template_partials_are_included(self, rendered: str):
        assert self.PARTIAL in rendered

    def test_check_runs_after_every_env(self, rendered: str):
        """The gate must run after the last loader ENV, or it sees a clean env and the bad value ships anyway."""
        lines = rendered.splitlines()
        check_line = next(i for i, ln in enumerate(lines) if "karotte check" in ln)
        last_loader_env = max(
            i
            for i, ln in enumerate(lines)
            if ln.startswith("ENV") or ln.strip().startswith(("LD_", "PATH="))
            if any(v in ln for v in ("LD_LIBRARY_PATH", "PATH="))
        )
        assert check_line > last_loader_env, (
            "karotte check must run after the last loader-path ENV"
        )
