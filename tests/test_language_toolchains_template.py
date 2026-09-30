"""Tests for the `language-toolchains` add-on template and its default-template
seams.

The template's own invariants (pinned URLs, sha256s, sealing table) live in the
template's `tests/test_toolchains.py` and run inside a generated environment;
what belongs here is what `create_env` produces: the opt-in must add the
machinery, and the default template must be untouched by the seams that make
the opt-in possible.
"""

import importlib.util
from pathlib import Path
from unittest.mock import patch

import pytest

from karotte.create_env import create_env


@pytest.fixture(scope="module")
def default_env(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("envs") / "default_env"
    with patch("karotte.create_env.subprocess.check_call"):
        create_env(out, templates=["default"])
    return out


@pytest.fixture(scope="module")
def toolchains_env(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("envs") / "toolchains_env"
    with patch("karotte.create_env.subprocess.check_call"):
        create_env(out, templates=["language-toolchains"])
    return out


def _load_toolchains_module(env_dir: Path):
    """Import the rendered `toolchains.py` the way its build half is run: as a
    file, with `toolchain_config.py` found beside it rather than on sys.path."""
    path = env_dir / "src" / "environment" / "toolchains.py"
    spec = importlib.util.spec_from_file_location("toolchains_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestDefaultSeams:
    def test_containerfile_has_no_builder_residue(self, default_env: Path):
        """The empty `builder_stages` block must leave the file starting at the
        same first line as before the seam existed."""
        containerfile = (default_env / "Containerfile").read_text()
        assert containerfile.startswith("FROM docker.io/library/amazonlinux")
        assert "toolchains" not in containerfile

    def test_containerfile_has_no_jinja_leftovers(self, default_env: Path):
        containerfile = (default_env / "Containerfile").read_text()
        assert "{%" not in containerfile
        assert "{{" not in containerfile

    def test_justfile_imports_addon_recipes_optionally(self, default_env: Path):
        """`import?` tolerates the file being absent, which is what makes the
        seam free for environments that never opt in."""
        justfile = (default_env / "justfile").read_text()
        assert "import? 'toolchains.just'" in justfile
        assert not (default_env / "toolchains.just").exists()

    def test_check_permissions_discovers_addon_check_modules(self, default_env: Path):
        """Add-on image checks are found by the `*_checks` naming convention,
        so the default template never names a specific add-on."""
        source = (default_env / "src/environment/check_permissions.py").read_text()
        assert "_checks" in source
        assert "iter_modules" in source
        assert "toolchain" not in source

    def test_the_managed_cpython_gets_a_stack_header_and_loses_its_write_bits(
        self, default_env: Path
    ):
        """uv's CPython ships without PT_GNU_STACK, which glibc reads as a
        request for rwx thread stacks; the image patches and freezes it, and
        an image check verifies both."""
        containerfile = (default_env / "Containerfile").read_text()
        assert "COPY scripts/clear_execstack.py" in containerfile
        assert "chmod 0555 $(readlink -f ${UV_PYTHON_INSTALL_DIR}" in containerfile
        assert (default_env / "scripts/clear_execstack.py").is_file()
        assert (default_env / "src/environment/interpreter_checks.py").is_file()

    def test_clear_execstack_is_copied_right_before_each_step_that_runs_it(
        self, default_env: Path
    ):
        """Env build steps that end in `rm -rf /tmp/*` must not be able to
        delete the script between its COPY and a later RUN."""
        lines = (default_env / "Containerfile").read_text().splitlines()
        text = "\n".join(line for line in lines if not line.lstrip().startswith("#"))
        instructions = [
            line.split(maxsplit=1)
            for line in text.replace("\\\n", " ").splitlines()
            if line.strip()
        ]
        users = [
            i
            for i, (keyword, *rest) in enumerate(instructions)
            if keyword == "RUN" and "/tmp/clear_execstack.py" in " ".join(rest)
        ]
        assert len(users) == 2
        for i in users:
            steps_before = []
            for keyword, *rest in reversed(instructions[:i]):
                if keyword == "RUN" or "clear_execstack.py" in " ".join(rest):
                    steps_before.append((keyword, " ".join(rest)))
                    break
            assert steps_before and steps_before[0][0] == "COPY", (
                f"a RUN sits between the COPY and instruction {i}"
            )


class TestLanguageToolchainsTemplate:
    def test_manifest_records_both_templates(self, toolchains_env: Path):
        manifest = (toolchains_env / ".manifest.json").read_text()
        assert '"default"' in manifest
        assert '"language-toolchains"' in manifest

    def test_containerfile_gains_builder_stage_and_keeps_default(
        self, toolchains_env: Path
    ):
        containerfile = (toolchains_env / "Containerfile").read_text()
        assert " AS toolchains" in containerfile.splitlines()[0] or (
            "AS toolchains" in containerfile.split("FROM")[1]
        )
        assert "COPY --from=toolchains /opt/toolchains/go" in containerfile
        # The default template's own steps must survive the extension.
        assert "karotte check" in containerfile
        assert "useradd -M -d ${STUDENT_WORKDIR}" in containerfile

    def test_each_language_is_copied_out_on_its_own_line(self, toolchains_env: Path):
        """One layer per language: a single copy of the whole tree is one
        multi-gigabyte blob for a registry to gzip on one core, and one bumped
        version invalidates every other language's cache."""
        module = _load_toolchains_module(toolchains_env)
        containerfile = (toolchains_env / "Containerfile").read_text()

        for language in module.Language:
            directory = f"/opt/toolchains/{language.value}"
            assert f"COPY --from=toolchains {directory} {directory}\n" in containerfile
        assert "COPY --from=toolchains /opt/toolchains /opt/toolchains" not in (
            containerfile
        )

    def test_containerfile_sets_a_locale(self, toolchains_env: Path):
        """Ruby, the BEAM and the JVM take their default encoding from the
        locale, and the base image sets none: without this line a shell has
        whatever locale its launcher happened to leak in."""
        containerfile = (toolchains_env / "Containerfile").read_text()
        assert "LC_CTYPE=C.UTF-8" in containerfile

    def test_rpm_staging_runs_in_final_image(self, toolchains_env: Path):
        """`dnf download --resolve` must resolve against the final image's rpm
        database, so the staging step has to be in the final stage."""
        containerfile = (toolchains_env / "Containerfile").read_text()
        final_stage = containerfile.rsplit("COPY --from=toolchains", 1)[1]
        assert "rpms" in final_stage

    def test_machinery_files_are_created(self, toolchains_env: Path):
        for relative in (
            "src/environment/sandbox.py",
            "src/environment/toolchains.py",
            "src/environment/toolchain_config.py",
            "src/environment/toolchain_grading.py",
            "src/environment/toolchain_checks.py",
            "scripts/check_toolchains.py",
            "toolchains.just",
            "tests/test_toolchains.py",
            "tests/test_sandbox.py",
        ):
            assert (toolchains_env / relative).is_file(), f"missing {relative}"

    def test_no_language_enabled_by_default(self, toolchains_env: Path):
        module = _load_toolchains_module(toolchains_env)
        assert module.enabled_languages() == frozenset()

    def test_table_covers_every_language(self, toolchains_env: Path):
        module = _load_toolchains_module(toolchains_env)
        assert set(module.TOOLCHAINS) == set(module.Language)

    def test_table_pins_archives_for_both_architectures(self, toolchains_env: Path):
        module = _load_toolchains_module(toolchains_env)
        for toolchain in module.TOOLCHAINS.values():
            for archive in toolchain.archives:
                for architecture in ("x86_64", "aarch64"):
                    assert archive.urls[architecture].startswith("https://")
                    assert len(archive.sha256[architecture]) == 64

    def test_bad_config_is_rejected(self, toolchains_env: Path):
        """A typo'd language name must fail loudly, not install nothing."""
        module = _load_toolchains_module(toolchains_env)
        with pytest.raises(ValueError, match="rustt"):
            module.validate_languages(frozenset({"rustt"}))
