"""Tests for the language toolchain table.

Installing a toolchain needs a container and a network, so what is checkable
here is the table and the pure halves of the machinery. The build itself is
checked inside the image: `karotte check` runs the toolchain permission checks,
and `just check-toolchains` builds a hello-world per enabled language.
"""

import os
import re
import subprocess
import threading
from pathlib import Path

import pytest

from environment import STUDENT_UID, toolchain_grading, toolchains
from environment.paths import STUDENT_WORKDIR
from environment.toolchains import (
    TOOLCHAINS,
    TOOLCHAINS_DIR,
    Language,
    Toolchain,
    enabled_languages,
    toolchain_dir,
    validate_languages,
)

ARCHITECTURES = ("x86_64", "aarch64")


@pytest.fixture(params=list(Language), ids=lambda language: language.value)
def toolchain(request) -> Toolchain:
    return TOOLCHAINS[request.param]


@pytest.fixture
def fake_frontends(tmp_path, monkeypatch) -> Path:
    """gcc's frontends under `tmp_path`. Any test that installs a toolchain
    needs these: the globs are absolute, so unpatched they reach the real
    `/usr/libexec/gcc` — which a build box has and cannot chmod."""
    frontend_dir = tmp_path / "usr/libexec/gcc/aarch64-amazon-linux/11"
    frontend_dir.mkdir(parents=True)
    for name in ("cc1", "cc1plus"):
        (frontend_dir / name).touch(mode=0o755)
    relative = frontend_dir.relative_to("/")
    monkeypatch.setattr(
        toolchains,
        "C_FRONTEND_GLOBS",
        (f"{relative}/cc1", f"{relative}/cc1plus"),
    )
    return frontend_dir


class TestConfig:
    def test_the_config_names_only_real_languages(self):
        """Whatever the environment enabled, it must parse — this is the same
        validation the image build runs."""
        enabled_languages()

    def test_a_typo_fails_loudly(self):
        with pytest.raises(ValueError, match="rustt"):
            validate_languages(frozenset({"rustt"}))

    def test_every_language_value_is_accepted(self):
        values = frozenset(language.value for language in Language)
        assert validate_languages(values) == frozenset(Language)


class TestTable:
    def test_every_language_has_a_toolchain(self):
        assert set(TOOLCHAINS) == set(Language)

    def test_each_entry_names_its_own_language(self, toolchain: Toolchain):
        assert TOOLCHAINS[toolchain.language] is toolchain

    # The two cells that add nothing to the image, for different reasons:
    # Python's interpreter is already what the image runs on, and Assembly's
    # whole toolchain is the binutils the base image carries for `strings` and
    # `objdump`.
    INSTALLS_NOTHING = {Language.PYTHON, Language.ASSEMBLY}

    def test_every_other_language_installs_something(self, toolchain: Toolchain):
        installs = bool(toolchain.rpm_packages) or bool(toolchain.archives)
        assert installs is (toolchain.language not in self.INSTALLS_NOTHING)

    def test_an_archive_is_pinned_for_both_architectures(self, toolchain: Toolchain):
        if not toolchain.archives:
            pytest.skip(f"{toolchain.language} has no archive")
        for archive in toolchain.archives:
            for architecture in ARCHITECTURES:
                assert archive.urls.get(architecture)
                assert len(archive.sha256.get(architecture, "")) == 64

    def test_archives_are_fetched_over_https(self, toolchain: Toolchain):
        for archive in toolchain.archives:
            for url in archive.urls.values():
                assert url.startswith("https://")

    def test_distinct_architectures_get_distinct_bytes(self, toolchain: Toolchain):
        """Per archive: distinct URLs must pin distinct bytes, one URL one hash.

        Not per language, because a cell can mix both kinds: Mojo's compiler is
        a per-architecture binary while its stdlib is bytecode. The one-URL
        archives are JVM/JS bytecode, source the builder compiles in place
        (OTP, GnuCOBOL, GNU Prolog), and Mojo's stdlib.
        """
        if not toolchain.archives:
            pytest.skip(f"{toolchain.language} has no archive")
        for archive in toolchain.archives:
            urls = {archive.urls[a] for a in ARCHITECTURES}
            hashes = {archive.sha256[a] for a in ARCHITECTURES}
            assert len(hashes) == len(urls)

    def test_an_archive_says_where_its_binaries_are(self, toolchain: Toolchain):
        assert bool(toolchain.bin_dirs) is bool(toolchain.archives)

    def test_a_named_installer_exists(self, toolchain: Toolchain):
        for archive in toolchain.archives:
            if archive.installer is not None:
                assert archive.installer in toolchains.INSTALLERS

    def test_two_archives_in_one_cell_land_in_different_places(
        self, toolchain: Toolchain
    ):
        destinations = [archive.subdir for archive in toolchain.archives]
        assert len(set(destinations)) == len(destinations)


class TestCCompilerGrouping:
    """Ten languages cannot be installed without a C compiler coming too. That
    is accepted, but it should be visible in the table rather than a surprise at
    runtime."""

    NEEDS_GCC = {
        Language.C_CPP,
        Language.RUST,
        Language.OCAML,
        Language.HASKELL,
        Language.SWIFT,
        # Not to compile with — the .NET SDK hands the native-AOT link to a
        # platform linker, and gcc is the one this cell stages for it. Mojo,
        # GnuCOBOL and GNU Prolog stage it the same way: each drives the C
        # compiler for its final link (GnuCOBOL compiles through C outright).
        Language.CSHARP,
        Language.MOJO,
        Language.COBOL,
        Language.PROLOG,
        # gfortran is a gcc frontend; the RPM requires gcc either way, and the
        # table names it so the grouping is visible.
        Language.FORTRAN,
    }

    def test_the_languages_that_pull_gcc_are_the_ones_expected(self):
        pulls_gcc = {
            language
            for language, toolchain in TOOLCHAINS.items()
            if "gcc" in toolchain.rpm_packages
        }
        assert pulls_gcc == self.NEEDS_GCC

    @pytest.mark.parametrize(
        "language",
        [
            Language.GO,
            Language.DART,
            Language.KOTLIN,
            Language.JAVA,
            Language.SCALA,
            Language.JS_TS,
            Language.RUBY,
            Language.ERLANG_ELIXIR,
            Language.JULIA,
            Language.PYTHON,
            Language.LLVM_IR,
            Language.CLOJURE,
        ],
    )
    def test_the_clean_group_installs_no_compiler(self, language: Language):
        packages = TOOLCHAINS[language].rpm_packages
        assert not any("gcc" in p or "clang" in p for p in packages)


class TestCFrontend:
    """gcc arrives in most of the cells that stage it purely as the assembler
    and linker driver their compiler builds through. The driver compiles
    nothing itself — `cc1` and `cc1plus` do — so those are sealed at install
    everywhere the cell's own language does not need them."""

    KEEPS_IT = {Language.C_CPP, Language.COBOL}

    def test_the_cells_that_keep_it_are_the_ones_expected(self):
        keeps = {
            language
            for language, toolchain in TOOLCHAINS.items()
            if toolchain.c_frontend
        }
        assert keeps == self.KEEPS_IT

    def test_only_a_cell_that_stages_gcc_can_keep_it(self, toolchain: Toolchain):
        if toolchain.c_frontend:
            assert "gcc" in toolchain.rpm_packages

    def test_the_mode_lands_on_both_frontends(self, fake_frontends: Path):
        changed = toolchains._mode_c_frontends(toolchains.SEALED_MODE)
        assert {path.name for path in changed} == {"cc1", "cc1plus"}
        assert all(
            path.stat().st_mode & 0o777 == toolchains.SEALED_MODE for path in changed
        )

        toolchains._mode_c_frontends(toolchains.OPEN_MODE)
        assert (fake_frontends / "cc1").stat().st_mode & 0o777 == toolchains.OPEN_MODE

    @pytest.fixture
    def installable(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setenv("KAROTTE_CONTAINERIZED", "1")
        monkeypatch.setattr(
            toolchains, "enabled_languages", lambda: frozenset(Language)
        )
        monkeypatch.setattr(toolchains, "TOOLCHAINS_DIR", tmp_path / "toolchains")
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        monkeypatch.setattr(toolchains, "SYSTEM_BIN_DIRS", (str(bin_dir),))
        monkeypatch.setattr(toolchains, "_install_rpms", lambda toolchain: None)
        monkeypatch.setattr(toolchains, "_write_wrappers", lambda toolchain: [])

    @pytest.mark.parametrize(
        ("language", "expected_mode"),
        [(Language.RUST, 0o700), (Language.C_CPP, toolchains.BUILDER_MODE)],
        ids=["sealed-for-a-driver-cell", "builder-only-where-c-is-the-language"],
    )
    def test_install_seals_them_unless_the_cell_compiles_c(
        self, fake_frontends, installable, language, expected_mode
    ):
        toolchains.install_toolchain(language)

        assert (fake_frontends / "cc1").stat().st_mode & 0o777 == expected_mode
        assert (fake_frontends / "cc1plus").stat().st_mode & 0o777 == expected_mode

    def test_the_table_outranks_the_rpm_gcc_stages_to_link(
        self, fake_frontends, installable, monkeypatch
    ):
        """A cell that stages gcc lists the frontends among its RPM
        executables, and that grant must not reopen what the table sealed."""
        monkeypatch.setattr(
            toolchains,
            "_rpm_executables",
            lambda toolchain: [fake_frontends / "cc1", fake_frontends / "cc1plus"],
        )

        toolchains.install_toolchain(Language.HASKELL)

        for name in ("cc1", "cc1plus"):
            mode = (fake_frontends / name).stat().st_mode & 0o777
            assert mode == toolchains.SEALED_MODE


class TestHaskellMainStub:
    """GHC compiles a generated C `main` stub at every link — the one thing a
    plain Haskell build needs the sealed C frontend for. The builder compiles
    the stub once, and a build links `-no-hs-main` with it instead."""

    def test_the_stub_lives_in_the_haskell_toolchain(self):
        stub = Path(toolchains.HASKELL_MAIN_STUB)
        assert toolchain_dir(Language.HASKELL) in stub.parents

    def test_the_haskell_cell_does_not_keep_the_frontend(self):
        assert not TOOLCHAINS[Language.HASKELL].c_frontend


class TestBaseImageBuildTools:
    """`as` and `ld` are in every cell's image whether or not the cell compiles,
    and a submission can carry a prebuilt object as bytes. So the assembler is
    handed out per cell, the way a language directory is."""

    # The two backends whose builds name them directly, Pascal — whose compiler
    # has no C frontend and drives both itself — and every cell that stages
    # gcc, which drives them out of sight.
    NEEDS_BINUTILS = {
        Language.ASSEMBLY,
        Language.LLVM_IR,
        Language.PASCAL,
    } | TestCCompilerGrouping.NEEDS_GCC

    def test_the_cells_that_get_them_are_the_ones_expected(self):
        opened = {
            language for language, toolchain in TOOLCHAINS.items() if toolchain.binutils
        }
        assert opened == self.NEEDS_BINUTILS

    def test_a_cell_that_stages_gcc_gets_them(self, toolchain: Toolchain):
        # gcc assembles and links through the base image's binutils; without
        # them the cell cannot compile at all.
        if "gcc" in toolchain.rpm_packages:
            assert toolchain.binutils

    def test_the_mode_lands_on_every_name_the_image_has(self, tmp_path, monkeypatch):
        monkeypatch.setattr(toolchains, "SYSTEM_BIN_DIRS", (str(tmp_path),))
        for name in ("as", "ld"):
            (tmp_path / name).touch(mode=0o755)

        changed = toolchains._mode_base_image_build_tools(toolchains.SEALED_MODE)
        assert {path.name for path in changed} == {"as", "ld"}
        assert all(
            path.stat().st_mode & 0o777 == toolchains.SEALED_MODE for path in changed
        )

        toolchains._mode_base_image_build_tools(toolchains.OPEN_MODE)
        assert (tmp_path / "as").stat().st_mode & 0o777 == toolchains.OPEN_MODE

    def test_the_build_half_closes_them_before_any_cell_opens_one(
        self, tmp_path, monkeypatch
    ):
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        (bin_dir / "as").touch(mode=0o755)
        monkeypatch.setenv("KAROTTE_CONTAINERIZED", "1")
        monkeypatch.setattr(toolchains, "SYSTEM_BIN_DIRS", (str(bin_dir),))
        monkeypatch.setattr(toolchains, "TOOLCHAINS_DIR", tmp_path / "toolchains")
        toolchains.TOOLCHAINS_DIR.mkdir()

        toolchains._seal()
        assert (bin_dir / "as").stat().st_mode & 0o777 == toolchains.SEALED_MODE


class TestBuilderInstall:
    """`install_toolchain` hands the toolchain to the builder uid and nothing
    to the student: the build MCP tool compiles on the student's behalf, and
    only what `keep` names — the artifacts' own runtime — stays theirs."""

    @pytest.fixture
    def subordinate_ids(self, tmp_path, monkeypatch) -> Path:
        allocations = tmp_path / "subuid"
        allocations.write_text("student:100000:65536\nbuilder:165536:65536\n")
        monkeypatch.setattr(toolchains, "SUBORDINATE_ID_FILES", (allocations,))
        return allocations

    @pytest.fixture
    def bin_dir(self, tmp_path, monkeypatch, fake_frontends, subordinate_ids) -> Path:
        monkeypatch.setenv("KAROTTE_CONTAINERIZED", "1")
        monkeypatch.setattr(
            toolchains, "enabled_languages", lambda: frozenset(Language)
        )
        monkeypatch.setattr(toolchains, "TOOLCHAINS_DIR", tmp_path / "toolchains")
        wrapper_dir = tmp_path / "wrappers"
        wrapper_dir.mkdir()
        monkeypatch.setattr(toolchains, "WRAPPER_DIR", wrapper_dir)
        monkeypatch.setattr(toolchains, "_install_rpms", lambda toolchain: None)
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        monkeypatch.setattr(toolchains, "SYSTEM_BIN_DIRS", (str(bin_dir),))
        return bin_dir

    def test_nothing_it_installs_lands_outside_the_temporary_tree(
        self, bin_dir, tmp_path, fake_frontends
    ):
        """The modes below are real chmods. Every path one can reach has to be
        under `tmp_path`, or a run on a box with gcc rewrites the box."""
        toolchains.install_toolchain(Language.GO)

        for frontend in toolchains.c_frontends():
            assert tmp_path in frontend.parents
        assert (fake_frontends / "cc1").stat().st_mode & 0o777 == toolchains.SEALED_MODE

    def test_the_builder_mode_shuts_the_world_out(self):
        assert toolchains.BUILDER_MODE & 0o007 == 0
        assert toolchains.BUILDER_MODE & 0o050 == 0o050

    def test_the_builder_is_not_the_student(self):
        assert toolchains.BUILDER_UID != STUDENT_UID

    def test_a_dir_sealed_cell_grants_the_directory_to_the_builder(self, bin_dir):
        directory = toolchain_dir(Language.GO)
        (directory / "bin").mkdir(parents=True)
        (directory / "bin" / "go").touch(mode=0o755)

        toolchains.install_toolchain(Language.GO)

        assert directory.stat().st_mode & 0o777 == toolchains.BUILDER_MODE

    def test_a_file_by_file_cell_keeps_the_directory_open(self, bin_dir):
        """The BEAM cell's kept executable is `erts/bin/beam.smp`, which is in
        no `bin_dir` — so the walk that decides modes has to be over the whole
        tree, not over the wrappers' directories."""
        directory = toolchain_dir(Language.ERLANG_ELIXIR)
        (directory / "erts" / "bin").mkdir(parents=True)
        (directory / "erts" / "bin" / "beam.smp").touch(mode=0o755)
        (directory / "bin").mkdir(parents=True)
        (directory / "bin" / "erlc").touch(mode=0o755)

        toolchains.install_toolchain(Language.ERLANG_ELIXIR)

        assert directory.stat().st_mode & 0o777 == toolchains.OPEN_MODE
        kept = directory / "erts" / "bin" / "beam.smp"
        assert kept.stat().st_mode & 0o777 == 0o755
        compiler = directory / "bin" / "erlc"
        assert compiler.stat().st_mode & 0o777 == toolchains.BUILDER_MODE

    def test_rpm_executables_go_to_the_builder_except_keep(self, bin_dir, monkeypatch):
        ruby = bin_dir / "ruby"
        erb = bin_dir / "erb"
        for path in (ruby, erb):
            path.touch(mode=0o755)
        monkeypatch.setattr(
            toolchains, "_rpm_executables", lambda toolchain: [ruby, erb]
        )

        toolchains.install_toolchain(Language.RUBY)

        assert ruby.stat().st_mode & 0o777 == 0o755
        assert erb.stat().st_mode & 0o777 == toolchains.BUILDER_MODE

    def test_the_distributions_jvm_is_the_builders_alone(self, bin_dir, monkeypatch):
        """The JDK's own `java` carries a Java compiler as a module, so a JVM
        cell keeps none of it: what its run starts is the jlink'ed runtime in
        the cell's own directory."""
        java = bin_dir / "java"
        javac = bin_dir / "javac"
        for path in (java, javac):
            path.touch(mode=0o755)
        monkeypatch.setattr(
            toolchains, "_rpm_executables", lambda toolchain: [java, javac]
        )

        toolchains.install_toolchain(Language.JAVA)

        assert java.stat().st_mode & 0o777 == toolchains.BUILDER_MODE
        assert javac.stat().st_mode & 0o777 == toolchains.BUILDER_MODE

    @pytest.mark.parametrize(
        ("language", "expected_mode"),
        [(Language.RUST, toolchains.BUILDER_MODE), (Language.GO, 0o700)],
        ids=["builder-only-where-the-cell-links-through-them", "sealed-elsewhere"],
    )
    def test_the_base_image_tools_follow_the_binutils_flag(
        self, bin_dir, language, expected_mode
    ):
        assembler = bin_dir / "as"
        assembler.touch(mode=0o755)

        toolchains.install_toolchain(language)

        assert assembler.stat().st_mode & 0o777 == expected_mode

    def test_what_it_closes_is_shut_to_the_builder_too(self, bin_dir):
        """`install_toolchain` hands the language directory to the builder, so
        this has to land after that or the grant gives it straight back."""
        directory = toolchain_dir(Language.HASKELL)
        iserv = directory / "lib" / "ghc-9.8.4" / "lib" / "bin"
        iserv.mkdir(parents=True)
        (iserv / "ghc-iserv-ghc-9.8.4").touch(mode=0o755)
        (iserv / "unlit").touch(mode=0o755)

        toolchains.install_toolchain(Language.HASKELL)

        splices = iserv / "ghc-iserv-ghc-9.8.4"
        assert splices.stat().st_mode & 0o777 == toolchains.SEALED_MODE
        assert splices.stat().st_mode & 0o077 == 0
        assert (iserv / "unlit").stat().st_mode & 0o777 == 0o755

    def test_it_closes_the_file_a_link_points_at(self, bin_dir):
        """The distribution ships `ghc-iserv` as a link to a versioned name,
        and a mode on the link would be a mode on nothing."""
        directory = toolchain_dir(Language.HASKELL)
        iserv = directory / "lib" / "ghc-9.8.4" / "lib" / "bin"
        iserv.mkdir(parents=True)
        real = iserv / "ghc-iserv-ghc-9.8.4"
        real.touch(mode=0o755)
        (iserv / "ghc-iserv").symlink_to(real.name)

        toolchains.install_toolchain(Language.HASKELL)

        assert real.stat().st_mode & 0o777 == toolchains.SEALED_MODE

    def test_wrappers_are_builder_only_unless_kept(self, bin_dir):
        directory = toolchain_dir(Language.JULIA)
        (directory / "bin").mkdir(parents=True)
        (directory / "bin" / "julia").touch(mode=0o755)
        (directory / "bin" / "dsymutil").touch(mode=0o755)

        toolchains.install_toolchain(Language.JULIA)

        wrappers = toolchains.WRAPPER_DIR
        assert (wrappers / "julia").stat().st_mode & 0o777 == 0o755
        assert (wrappers / "dsymutil").stat().st_mode & 0o777 == toolchains.BUILDER_MODE

    def test_the_image_build_creates_the_builder_account(
        self, bin_dir, subordinate_ids, monkeypatch
    ):
        commands: list[tuple[str, ...]] = []
        monkeypatch.setattr(
            toolchains, "_run", lambda *args, **kwargs: commands.append(args)
        )
        monkeypatch.setattr(toolchains, "_stage_rpms", lambda toolchain: None)
        toolchains.TOOLCHAINS_DIR.mkdir(parents=True, exist_ok=True)

        toolchains.stage_rpms_and_seal()

        assert ("groupadd", "--gid", str(toolchains.BUILDER_UID), "builder") in commands
        useradd = next(args for args in commands if args[0] == "useradd")
        assert str(toolchains.BUILDER_UID) in useradd
        assert "builder" in useradd

    def test_the_builder_gets_no_subordinate_ids_to_remap_with(
        self, bin_dir, subordinate_ids, monkeypatch
    ):
        monkeypatch.setattr(toolchains, "_run", lambda *args, **kwargs: None)
        monkeypatch.setattr(toolchains, "_stage_rpms", lambda toolchain: None)
        toolchains.TOOLCHAINS_DIR.mkdir(parents=True, exist_ok=True)

        toolchains.stage_rpms_and_seal()

        assert subordinate_ids.read_text() == "student:100000:65536\n"


class TestBaseImageInterpreters:
    def test_the_ones_that_arrive_uninvited_are_on_the_list(self):
        # gawk ships with the base image; perl arrives as an RPM dependency
        # (`perf` pulls perl-interpreter in).
        assert {"perl*", "gawk", "awk"} <= set(toolchains.BASE_IMAGE_INTERPRETER_GLOBS)

    def test_sealing_closes_every_name_the_image_has(self, tmp_path, monkeypatch):
        monkeypatch.setattr(toolchains, "SYSTEM_BIN_DIRS", (str(tmp_path),))
        for name in ("perl", "perl5.32.1", "gawk"):
            (tmp_path / name).touch(mode=0o755)
        (tmp_path / "awk").symlink_to(tmp_path / "gawk")
        (tmp_path / "sed").touch(mode=0o755)

        closed = toolchains._seal_base_image_interpreters(frozenset())
        assert {path.name for path in closed} == {"perl", "perl5.32.1", "gawk", "awk"}
        for name in ("perl", "perl5.32.1", "gawk"):
            assert (tmp_path / name).stat().st_mode & 0o777 == toolchains.SEALED_MODE
        assert (tmp_path / "sed").stat().st_mode & 0o777 == 0o755

    def test_a_kept_interpreter_stays_open(self, tmp_path, monkeypatch):
        monkeypatch.setattr(toolchains, "SYSTEM_BIN_DIRS", (str(tmp_path),))
        for name in ("ruby", "perl"):
            (tmp_path / name).touch(mode=0o755)

        closed = toolchains._seal_base_image_interpreters(frozenset({"ruby"}))
        assert {path.name for path in closed} == {"perl"}
        assert (tmp_path / "ruby").stat().st_mode & 0o777 == 0o755

    def test_the_interpreted_cells_keep_theirs_by_the_names_the_globs_close(self):
        # If these drift apart, sealing takes the cell's own runtime away.
        assert "ruby" in TOOLCHAINS[Language.RUBY].keep
        assert "node" in TOOLCHAINS[Language.JS_TS].keep


class TestJvmCells:
    """A JVM cell runs its jar on a runtime of its own rather than on the
    distribution's JDK, because that JDK carries a Java compiler as a module —
    `jdk.compiler` — which a submission reaches in process, with no writable
    file and no syscall the filter sees. Linking the runtime without it is the
    only thing that takes it away."""

    LINKS_ITS_OWN = {
        Language.KOTLIN,
        Language.JAVA,
        Language.SCALA,
        Language.CLOJURE,
    }

    def test_the_cells_that_link_one_are_the_ones_expected(self):
        links = {
            language for language, toolchain in TOOLCHAINS.items() if toolchain.run_tree
        }

        assert links == self.LINKS_ITS_OWN

    def test_none_of_them_keeps_the_distributions_jvm(self, toolchain: Toolchain):
        """Keeping it by name would keep the compiler with it, and `keep`
        matches basenames, so both `java`s would survive as one."""
        if toolchain.language in self.LINKS_ITS_OWN:
            assert toolchain.keep == ()

    def test_the_run_starts_the_linked_runtime(self, toolchain: Toolchain):
        if toolchain.language in self.LINKS_ITS_OWN:
            java = toolchains.jvm_java(toolchain.language)

            assert java == toolchain_dir(toolchain.language) / "jre" / "bin" / "java"

    def test_the_linked_runtime_has_no_compiler_in_it(self):
        """The alarm for the day someone adds a module back for convenience."""
        for module in ("jdk.compiler", "jdk.jshell", "java.compiler"):
            assert module not in toolchains.JRE_MODULES

    def test_each_run_is_denied_native_access(self, toolchain: Toolchain):
        """Removing the compiler is not enough: FFM reaches native code, which
        the JIT shim cannot refuse. `--illegal-native-access=deny` closes it,
        and only a JDK new enough to have it can — hence the version floor."""
        if toolchain.language in self.LINKS_ITS_OWN:
            assert "--illegal-native-access=deny" in toolchain.run_flags

    def test_the_jdk_is_new_enough_to_deny_native_access(self):
        """`--illegal-native-access=deny` is an LTS behaviour from JDK 24 on;
        the compiler-free runtime alone (any JDK) would still leave FFM open."""
        assert int(toolchains.JVM_JDK) >= 24

    def test_the_confined_cells_share_one_jdk(self):
        """The builder stage links every JVM cell's runtime from one set of
        jmods; a cell on a different major would jlink against the wrong ones."""
        majors = {
            rpm.split("-")[1]
            for language in self.LINKS_ITS_OWN
            for rpm in TOOLCHAINS[language].rpm_packages
            if "corretto" in rpm
        }
        assert majors == {toolchains.JVM_JDK}

    @pytest.fixture
    def sealable(self, tmp_path, monkeypatch, fake_frontends) -> None:
        """Every path installing and sealing would chmod, pointed at
        `tmp_path`: on a dev box they are the host's own binaries."""
        monkeypatch.setenv("KAROTTE_CONTAINERIZED", "1")
        monkeypatch.setattr(
            toolchains, "enabled_languages", lambda: frozenset(Language)
        )
        monkeypatch.setattr(toolchains, "TOOLCHAINS_DIR", tmp_path / "toolchains")
        monkeypatch.setattr(toolchains, "UV_PYTHON_DIR", tmp_path / "uv")
        monkeypatch.setattr(toolchains, "PYTHON_GLOBS", ())
        monkeypatch.setattr(toolchains, "_install_rpms", lambda toolchain: None)
        monkeypatch.setattr(toolchains, "_rpm_executables", lambda toolchain: [])
        # File-by-file sealing hands each file to root first, which only the
        # image is; here the mode is the whole of what is being checked.
        monkeypatch.setattr(toolchains.os, "chown", lambda *args: None)
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        monkeypatch.setattr(toolchains, "SYSTEM_BIN_DIRS", (str(bin_dir),))
        monkeypatch.setattr(toolchains, "WRAPPER_DIR", bin_dir)

    def test_sealing_leaves_the_run_tree_alone(self, sealable):
        """It holds the runtime the scored run starts, and file-by-file sealing
        would otherwise close its `java` along with the compiler's."""
        directory = toolchain_dir(Language.KOTLIN)
        (directory / "bin").mkdir(parents=True)
        (directory / "bin" / "kotlinc").touch(mode=0o755)
        (directory / "jre" / "bin").mkdir(parents=True)
        (directory / "jre" / "bin" / "java").touch(mode=0o755)

        toolchains.install_toolchain(Language.KOTLIN)
        toolchains.seal_toolchain(Language.KOTLIN)

        assert (directory / "jre" / "bin" / "java").stat().st_mode & 0o777 == 0o755
        assert (directory / "bin" / "kotlinc").stat().st_mode & 0o777 == (
            toolchains.SEALED_MODE
        )


def student_readable(root: Path) -> list[Path]:
    """Every file under `root` a uid that owns none of it could still read:
    o+r on the file, and o+x on `root` and every directory between."""
    if not root.stat().st_mode & 0o001:
        return []
    readable: list[Path] = []
    for path in sorted(root.iterdir()):
        if path.is_dir():
            readable.extend(student_readable(path))
        elif path.stat().st_mode & 0o004:
            readable.append(path)
    return readable


class TestManagedPythonSealing:
    """Sealing `bin/python3` is not sealing Python: the uv-managed CPython's
    `libpython3.*.so` exports `Py_Initialize`, so a compiled artifact that can
    dlopen it and read the stdlib beside it runs Python in any cell."""

    @pytest.fixture
    def uv_python(self, tmp_path, monkeypatch) -> Path:
        """A stand-in for the uv-managed CPython, at a path the tests may chmod."""
        root = tmp_path / "opt/uv/python"
        install = root / "cpython-3.12.11-linux-x86_64-gnu"
        stdlib = install / "lib" / "python3.12"
        (install / "bin").mkdir(parents=True)
        (stdlib / "lib-dynload").mkdir(parents=True)
        (install / "bin" / "python3").touch(mode=0o755)
        (install / "lib" / "libpython3.12.so.1.0").touch(mode=0o755)
        (stdlib / "os.py").touch(mode=0o644)
        (stdlib / "lib-dynload" / "unicodedata.so").touch(mode=0o755)
        monkeypatch.setattr(toolchains, "UV_PYTHON_DIR", root)
        monkeypatch.setattr(
            toolchains,
            "PYTHON_GLOBS",
            (str((install / "bin" / "python*").relative_to("/")),),
        )
        return root

    @pytest.fixture
    def sealable(self, tmp_path, monkeypatch) -> None:
        """Everything else `seal_toolchain` chmods, pointed at `tmp_path`."""
        monkeypatch.setenv("KAROTTE_CONTAINERIZED", "1")
        monkeypatch.setattr(toolchains, "TOOLCHAINS_DIR", tmp_path / "toolchains")
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        monkeypatch.setattr(toolchains, "SYSTEM_BIN_DIRS", (str(bin_dir),))
        monkeypatch.setattr(toolchains, "WRAPPER_DIR", bin_dir)

    def test_nothing_under_it_is_student_readable_after_sealing(
        self, uv_python, sealable
    ):
        toolchains.seal_toolchain(Language.GO)

        assert student_readable(uv_python) == []

    def test_the_library_and_stdlib_go_with_the_binary(self, uv_python, sealable):
        """The reported way around the old sealing: closing the executables
        alone left a whole interpreter behind as a shared library."""
        closed = toolchains.seal_toolchain(Language.GO)

        assert uv_python in closed
        assert uv_python.stat().st_mode & 0o777 == toolchains.SEALED_MODE

    def test_root_can_still_reach_the_interpreter_it_grades_with(
        self, uv_python, sealable
    ):
        """The grader runs the managed CPython for its own baselines after
        sealing, and root traverses a 0700 directory regardless — so what must
        hold is that the tree is still there, whole."""
        toolchains.seal_toolchain(Language.GO)

        install = next(uv_python.iterdir())
        assert (install / "bin" / "python3").is_file()
        assert (install / "lib" / "libpython3.12.so.1.0").is_file()

    def test_the_python_cell_keeps_the_whole_tree(
        self, uv_python, sealable, monkeypatch
    ):
        """Its submission is run with that interpreter after sealing, which
        needs the library and stdlib as much as the binary."""
        interpreter = next(uv_python.iterdir()) / "bin" / "python3"
        monkeypatch.setattr(toolchains, "student_python", lambda: interpreter)

        toolchains.seal_toolchain(Language.PYTHON, keep_student_python=True)

        assert student_readable(uv_python)
        assert interpreter.stat().st_mode & 0o777 == toolchains.READ_ONLY_MODE

    def test_the_interpreter_the_python_cell_keeps_cannot_be_written_over(
        self, uv_python, sealable, monkeypatch
    ):
        interpreter = next(uv_python.iterdir()) / "bin" / "python3"
        monkeypatch.setattr(toolchains, "student_python", lambda: interpreter)

        toolchains.seal_toolchain(Language.PYTHON, keep_student_python=True)

        assert not interpreter.stat().st_mode & 0o222
        assert os.access(interpreter, os.X_OK)


class TestSealing:
    def test_it_refuses_to_run_outside_the_image(self, monkeypatch):
        monkeypatch.delenv("KAROTTE_CONTAINERIZED", raising=False)
        with pytest.raises(RuntimeError, match="inside the environment image"):
            toolchains.seal_toolchain(Language.RUST)

    def test_installing_refuses_too(self, monkeypatch):
        # It decides the mode of /usr/bin/as, which on a dev box is the host's.
        monkeypatch.delenv("KAROTTE_CONTAINERIZED", raising=False)
        with pytest.raises(RuntimeError, match="inside the environment image"):
            toolchains.install_toolchain(Language.RUST)

    def test_the_build_half_refuses_too(self, monkeypatch):
        # Same paths, reached from the Containerfile's `rpms` phase.
        monkeypatch.delenv("KAROTTE_CONTAINERIZED", raising=False)
        with pytest.raises(RuntimeError, match="inside the environment image"):
            toolchains._seal()

    def test_installing_a_disabled_language_refuses(self, monkeypatch):
        # Nothing was staged for it, so opening it would half-work silently.
        # The enabled set is stubbed so the guard fires before any chmod, even
        # when this runs in an env whose config enables everything.
        monkeypatch.setenv("KAROTTE_CONTAINERIZED", "1")
        monkeypatch.setattr(toolchains, "enabled_languages", lambda: frozenset())
        with pytest.raises(RuntimeError, match="ENABLED_LANGUAGES"):
            toolchains.install_toolchain(Language.RUST)

    def test_the_base_image_tools_it_closes_are_the_ones_it_did_not_install(self):
        assert {"as", "ld"} <= set(toolchains.BASE_IMAGE_BUILD_TOOLS)

    def test_the_utilities_it_borrows_are_not_ones_that_compile(self):
        assert not set(toolchains.BASE_IMAGE_BUILD_UTILITIES) & set(
            toolchains.BASE_IMAGE_BUILD_TOOLS
        )

    @pytest.mark.parametrize(
        ("language", "installed", "expected"),
        [
            (Language.JULIA, ["julia", "lld"], True),
            (Language.ERLANG_ELIXIR, ["erl", "erlc", "escript"], False),
            # Kotlin keeps a `java`, but that came from an RPM — the archive
            # holds only the compiler, so closing the directory is still right.
            (Language.KOTLIN, ["kotlinc", "kotlin"], False),
            (Language.JS_TS, ["tsc", "tsserver"], False),
        ],
    )
    def test_whether_the_archive_seals_by_file_follows_from_keep(
        self, monkeypatch, language, installed, expected
    ):
        monkeypatch.setattr(
            toolchains,
            "_archive_executables",
            lambda toolchain: [toolchain_dir(language) / "bin" / n for n in installed],
        )
        assert (
            toolchains._archive_holds_something_kept(TOOLCHAINS[language]) is expected
        )

    def test_a_cell_can_demand_file_by_file_sealing_without_a_keep(self, monkeypatch):
        """Mojo's artifacts load shared libraries out of the archive at run
        time, so the directory has to stay open even though no executable in it
        is kept."""
        monkeypatch.setattr(
            toolchains,
            "_archive_executables",
            lambda toolchain: [toolchain_dir(Language.MOJO) / "bin" / "mojo"],
        )
        toolchain = TOOLCHAINS[Language.MOJO]
        assert toolchain.seal_by_file
        assert not toolchains._archive_holds_something_kept(toolchain)
        assert toolchains._seals_file_by_file(toolchain)

    def test_the_cells_that_close_something_to_the_builder_are_expected(self):
        """Only Haskell, whose `ghc-iserv` is what a Template Haskell splice
        would be evaluated in."""
        within = {
            language
            for language, toolchain in TOOLCHAINS.items()
            if toolchain.seal_within
        }
        assert within == {Language.HASKELL}

    def test_the_cells_that_seal_by_flag_are_the_ones_expected(self):
        """Mojo because its artifacts load libraries out of the archive, the
        BEAM cell because the one executable it keeps is not in a `bin_dir`."""
        by_flag = {
            language
            for language, toolchain in TOOLCHAINS.items()
            if toolchain.seal_by_file
        }
        assert by_flag == {Language.MOJO, Language.ERLANG_ELIXIR, Language.CLOJURE}

    def test_it_looks_for_them_where_the_image_puts_them(self):
        assert toolchains.WRAPPER_DIR == Path("/usr/local/bin")
        assert str(toolchains.WRAPPER_DIR) in toolchains.SYSTEM_BIN_DIRS


class TestWrapperEnvironment:
    """Whatever a grader sets around a build has to be on the student's `PATH`
    too, or the documented command fails for the student alone — and the
    workaround they reach for is one the grader never runs."""

    # Three compilers cannot find their own pieces unaided: fpc has no
    # /etc/fpc.cfg to fall back on, dotnet restores from nuget.org unless
    # pointed at the cache the installer filled, and mojo takes its driver and
    # stdlib paths from the MODULAR_* variables its own pip launcher would set.
    NEEDS_WRAPPER_ENV = {Language.PASCAL, Language.CSHARP, Language.MOJO}

    def test_the_cells_that_carry_one_are_the_ones_expected(self, toolchain: Toolchain):
        carries = bool(toolchain.wrapper_env)
        assert carries is (toolchain.language in self.NEEDS_WRAPPER_ENV)

    def test_a_wrapper_can_only_name_its_own_toolchain_directory(
        self, toolchain: Toolchain
    ):
        for value in toolchain.wrapper_env.values():
            assert set(re.findall(r"\{[^}]*\}", value)) <= {"{prefix}"}

    def test_there_are_wrappers_to_carry_it(self, toolchain: Toolchain):
        # Only archive executables get a wrapper; an RPM's binaries are the
        # distribution's own and go on `PATH` untouched.
        if toolchain.wrapper_env:
            assert toolchain.archives and toolchain.bin_dirs

    def test_the_wrapper_exports_before_it_execs(self, tmp_path, monkeypatch):
        monkeypatch.setattr(toolchains, "TOOLCHAINS_DIR", tmp_path / "toolchains")
        monkeypatch.setattr(toolchains, "WRAPPER_DIR", tmp_path / "bin")
        (tmp_path / "bin").mkdir()
        prefix = toolchains.TOOLCHAINS_DIR / Language.PASCAL.value
        (prefix / "bin").mkdir(parents=True)
        compiler = prefix / "bin" / "fpc"
        compiler.touch(mode=0o755)

        (wrapper,) = toolchains._write_wrappers(TOOLCHAINS[Language.PASCAL])
        script = wrapper.read_text()
        assert f'export PPC_CONFIG_PATH="{prefix}/etc"' in script
        assert script.index("export") < script.index("exec")
        assert f'exec "{compiler}"' in script


class TestRunEnvironment:
    """A cell whose run starts the runtime directly has no wrapper to export
    what the wrapper would have, so the table carries it instead."""

    NEEDS_RUN_ENV = {Language.ERLANG_ELIXIR}

    def test_the_cells_that_carry_one_are_the_ones_expected(self, toolchain: Toolchain):
        carries = bool(toolchain.run_env)
        assert carries is (toolchain.language in self.NEEDS_RUN_ENV)

    def test_it_can_only_name_its_own_toolchain_directory(self, toolchain: Toolchain):
        for value in toolchain.run_env.values():
            assert set(re.findall(r"\{[^}]*\}", value)) <= {"{prefix}"}

    @pytest.mark.parametrize("attribute", ["run_env", "wrapper_env"])
    def test_no_value_names_a_path_outside_the_toolchain(
        self, toolchain: Toolchain, attribute
    ):
        """A path written as an f-string would have to double its braces, and a
        doubled brace in a template .py is a Jinja expression `create_env`
        renders away — leaving a value that starts at the filesystem root."""
        for value in getattr(toolchain, attribute).values():
            assert not value.startswith("/")

    def test_the_scored_runs_environment_carries_it(self):
        env = toolchain_grading.sandboxed_env({}, Language.ERLANG_ELIXIR)
        prefix = toolchain_dir(Language.ERLANG_ELIXIR)

        otp_root = f"{prefix}/{toolchains.OTP_ROOT_SUBDIR}"
        assert env["BINDIR"] == f"{otp_root}/{toolchains.ERTS_SUBDIR}/bin"
        assert env["ERL_INETRC"] == f"{prefix}/{toolchains.ERLANG_INETRC}"
        assert env["ERL_CRASH_DUMP_SECONDS"] == "0"

    def test_a_cell_without_one_gets_the_confinement_and_nothing_else(self):
        assert set(toolchain_grading.sandboxed_env({}, Language.GO)) == {"LD_PRELOAD"}

    def test_it_does_not_displace_the_confinement(self):
        """`LD_PRELOAD` is what makes the run confined, so no table entry may
        end up overwriting it."""
        for toolchain in TOOLCHAINS.values():
            assert "LD_PRELOAD" not in toolchain.run_env


class TestTheRunPreload:
    """A library the run dlopens partway through is one the shim's filter
    refuses to map, so the loader has to have mapped it already."""

    NEEDS_RUN_PRELOAD = {Language.PASCAL}

    def test_the_cells_that_carry_one_are_the_ones_expected(self, toolchain: Toolchain):
        carries = bool(toolchain.run_preload)
        assert carries is (toolchain.language in self.NEEDS_RUN_PRELOAD)

    def test_every_entry_is_a_bare_soname(self, toolchain: Toolchain):
        """An absolute path would have to be right on both architectures, and
        the two disagree about where the libraries live."""
        for soname in toolchain.run_preload:
            assert "/" not in soname

    def test_the_sealing_leaves_every_entry_alone(self, toolchain: Toolchain):
        """`_rpm_executables` and `_archive_tree_executables` skip a resolved
        name with `.so` in it, which is what carries a preload through the
        seal that ends the build phase."""
        for soname in toolchain.run_preload:
            assert ".so" in soname

    def test_the_scored_run_loads_them_behind_the_shim(self):
        env = toolchain_grading.sandboxed_env({}, Language.PASCAL)

        assert env["LD_PRELOAD"].split(" ") == [
            str(toolchain_grading.SHIM),
            "libpthread.so.0",
            "libgcc_s.so.1",
        ]

    def test_a_cell_without_one_preloads_the_shim_alone(self):
        env = toolchain_grading.sandboxed_env({}, Language.GO)

        assert env["LD_PRELOAD"] == str(toolchain_grading.SHIM)


class TestTheBeamArgv:
    """`erl` and `elixir` are scripts that exec their way to `beam.smp`, and
    the shim refuses execve — so the run has to make that argv itself."""

    def test_it_starts_the_emulator_directly(self):
        argv = toolchains.beam_argv("/workdir")
        assert argv[0] == str(toolchains.beam_smp())
        assert argv[0].endswith(f"{toolchains.ERTS_SUBDIR}/bin/beam.smp")

    def test_it_passes_the_three_sections_erlexec_would_have(self):
        argv = toolchains.beam_argv("/workdir", "-noshell")
        root = toolchains.otp_root()

        assert argv.count("--") == 3
        assert argv[argv.index("-root") + 1] == str(root)
        assert argv[argv.index("-bindir") + 1] == f"{root}/{toolchains.ERTS_SUBDIR}/bin"
        assert argv[argv.index("-home") + 1] == "/workdir"
        assert argv[-1] == "-noshell"

    def test_the_root_it_names_is_the_one_make_install_writes_to(self):
        """`make install` lands the distribution under the prefix rather than
        over it, and an emulator handed the prefix finds no boot script."""
        root = toolchains.otp_root()
        prefix = toolchain_dir(Language.ERLANG_ELIXIR)

        assert root == prefix / toolchains.OTP_ROOT_SUBDIR
        assert root != prefix

    def test_the_bindir_it_names_is_the_one_the_run_environment_exports(self):
        argv = toolchains.beam_argv("/workdir")
        prefix = toolchain_dir(Language.ERLANG_ELIXIR)
        run_env = TOOLCHAINS[Language.ERLANG_ELIXIR].run_env

        assert argv[argv.index("-bindir") + 1] == run_env["BINDIR"].format(
            prefix=prefix
        )

    def test_an_elixir_run_boots_the_same_emulator_through_start_cli(self):
        argv = toolchains.elixir_argv("/workdir", "-e", "Server.main()")
        elixir_lib = toolchain_dir(Language.ERLANG_ELIXIR) / "elixir" / "lib"

        assert argv[0] == str(toolchains.beam_smp())
        assert argv[argv.index("-elixir_root") + 1] == str(elixir_lib)
        assert argv[argv.index("-pa") + 1] == str(elixir_lib / "elixir" / "ebin")
        # Everything after `-extra` is Elixir's own argv rather than the
        # emulator's, so a submission's arguments have to land past it.
        assert argv[argv.index("-extra") + 1 :] == ("-e", "Server.main()")


class TestPaths:
    def test_language_directories_are_under_the_toolchains_directory(
        self, toolchain: Toolchain
    ):
        directory = toolchain_dir(toolchain.language)
        assert directory.parent == TOOLCHAINS_DIR
        assert directory.name == toolchain.language.value

    def test_no_two_languages_share_a_directory(self):
        directories = {toolchain_dir(language) for language in Language}
        assert len(directories) == len(list(Language))

    def test_sealed_is_closed_and_open_is_not(self):
        assert toolchains.SEALED_MODE == 0o700
        assert toolchains.OPEN_MODE & 0o055


class TestDisplayNames:
    def test_every_language_has_one(self):
        for language in Language:
            assert language.display_name

    def test_they_are_distinct(self):
        names = {language.display_name for language in Language}
        assert len(names) == len(list(Language))


class TestBuildLocale:
    """Ruby, the BEAM and the JVM take their default encoding from the locale,
    and with none set they come up US-ASCII: the first multibyte literal fails
    under the grader alone. The build env is constructed from scratch, so the
    locale has to be put there deliberately."""

    def test_the_build_env_carries_it(self, tmp_path, monkeypatch):
        captured = {}

        def capture(argv, **kwargs):
            captured.update(kwargs)
            return subprocess.CompletedProcess(argv, returncode=1, stderr=b"stopped")

        monkeypatch.setattr(toolchain_grading.subprocess, "run", capture)
        submission = toolchain_grading.Submission(
            source="main.rs", build=(("rustc", "{source}"),)
        )
        source = tmp_path / "main.rs"
        source.write_bytes(b"fn main() {}")

        toolchain_grading.build_submission(submission, source, tmp_path)

        assert captured["env"] == {
            "PATH": toolchain_grading.SAFE_PATH,
            "HOME": str(tmp_path),
            "TMPDIR": str(tmp_path),
            "LC_CTYPE": toolchain_grading.UTF8_LOCALE,
        }

    def test_the_image_sets_the_same_one(self):
        """So nothing is true in a shell that is not true under the grader."""
        containerfile = Path(__file__).resolve().parent.parent / "Containerfile"

        assert f"LC_CTYPE={toolchain_grading.UTF8_LOCALE}" in containerfile.read_text()

    def test_a_compiler_that_answers_in_something_else_is_still_reported(
        self, tmp_path, monkeypatch
    ):
        """A compiler quoting the student's source back can put bytes that are
        not UTF-8 in front of the grader; a build failing is not a reason for
        the grader itself to raise."""

        def fail(argv, **kwargs):
            return subprocess.CompletedProcess(argv, returncode=1, stderr=b"bad \xff")

        monkeypatch.setattr(toolchain_grading.subprocess, "run", fail)
        submission = toolchain_grading.Submission(
            source="main.rs", build=(("rustc", "{source}"),)
        )
        source = tmp_path / "main.rs"
        source.write_bytes(b"fn main() {}")

        result = toolchain_grading.build_submission(submission, source, tmp_path)
        assert result.error is not None
        assert "bad �" in result.error


class TestStagingTheSubmission:
    """What the build directory holds before the first compiler runs."""

    def test_the_one_file_the_task_asked_for_is_what_lands(self, tmp_path, monkeypatch):
        workdir = tmp_path / "workdir"
        workdir.mkdir()
        (workdir / "main.rs").write_text("fn main() {}")
        monkeypatch.setattr(toolchain_grading, "STUDENT_WORKDIR", workdir)
        build_dir = tmp_path / "build"
        build_dir.mkdir()

        staged = toolchain_grading.stage_submission(
            toolchain_grading.Submission(source="main.rs"), build_dir
        )

        assert staged.read_text() == "fn main() {}"
        assert sorted(p.name for p in build_dir.iterdir()) == ["main.rs"]

    def test_nothing_links_back_out_to_the_students_workdir(
        self, tmp_path, monkeypatch
    ):
        """A link out is a path the build can follow to a second file, which
        is what staging one file is for."""
        workdir = tmp_path / "workdir"
        (workdir / "shared").mkdir(parents=True)
        (workdir / "main.rs").write_text("fn main() {}")
        monkeypatch.setattr(toolchain_grading, "STUDENT_WORKDIR", workdir)
        build_dir = tmp_path / "build"
        build_dir.mkdir()

        toolchain_grading.stage_submission(
            toolchain_grading.Submission(source="main.rs"), build_dir
        )

        assert not any(p.is_symlink() for p in build_dir.iterdir())


class TestBuildGuards:
    """What happens between the last build command exiting 0 and the build
    directory being handed back to root."""

    PRISTINE = b"fn main() {}"

    def _stage(self, tmp_path: Path) -> Path:
        source = tmp_path / "main.rs"
        source.write_bytes(self.PRISTINE)
        return source

    @staticmethod
    def _submission(**kwargs) -> "toolchain_grading.Submission":
        return toolchain_grading.Submission(
            source="main.rs", build=(("rustc", "{source}"),), **kwargs
        )

    @staticmethod
    def _build_does(monkeypatch, effect) -> None:
        """Make the build command a no-op with `effect` as its side effect."""

        def run(argv, **kwargs):
            effect()
            return subprocess.CompletedProcess(argv, returncode=0)

        monkeypatch.setattr(toolchain_grading.subprocess, "run", run)

    def test_processes_the_build_left_are_reaped_on_the_container(
        self, tmp_path, monkeypatch
    ):
        source = self._stage(tmp_path)
        self._build_does(monkeypatch, (tmp_path / "artifact").touch)
        monkeypatch.setenv("KAROTTE_CONTAINERIZED", "1")
        monkeypatch.setenv("KAROTTE_DEMOTE_ID", str(STUDENT_UID))
        reaped = []
        monkeypatch.setattr(toolchain_grading, "kill_processes", reaped.append)

        result = toolchain_grading.build_submission(
            self._submission(), source, tmp_path
        )

        assert result.error is None
        assert reaped == [toolchains.BUILDER_UID]

    def test_a_cohort_that_would_not_die_fails_the_build(self, tmp_path, monkeypatch):
        source = self._stage(tmp_path)
        self._build_does(monkeypatch, (tmp_path / "artifact").touch)
        monkeypatch.setenv("KAROTTE_CONTAINERIZED", "1")
        monkeypatch.setenv("KAROTTE_DEMOTE_ID", str(STUDENT_UID))

        def stuck(uid):
            raise RuntimeError("3 processes survived")

        monkeypatch.setattr(toolchain_grading, "kill_processes", stuck)

        result = toolchain_grading.build_submission(
            self._submission(), source, tmp_path
        )

        assert result.error is not None
        assert "3 processes survived" in result.error

    def test_reaping_is_skipped_off_the_container(self, tmp_path, monkeypatch):
        """Off the container nothing is demoted, and the builder uid may be a
        real user of the machine the tests run on."""
        source = self._stage(tmp_path)
        self._build_does(monkeypatch, (tmp_path / "artifact").touch)
        monkeypatch.delenv("KAROTTE_CONTAINERIZED", raising=False)

        def forbid(uid):
            raise AssertionError("reaped off the container")

        monkeypatch.setattr(toolchain_grading, "kill_processes", forbid)

        result = toolchain_grading.build_submission(
            self._submission(), source, tmp_path
        )

        assert result.error is None

    def test_a_symlink_left_at_the_artifact_fails_the_build(
        self, tmp_path, monkeypatch
    ):
        source = self._stage(tmp_path)
        # Pointed at a file that exists, so only the symlink guard can catch it.
        self._build_does(
            monkeypatch, lambda: (tmp_path / "artifact").symlink_to(source)
        )

        result = toolchain_grading.build_submission(
            self._submission(), source, tmp_path
        )

        assert result.error is not None
        assert "symlink" in result.error

    def test_scratch_beside_the_artifact_is_cleared(self, tmp_path, monkeypatch):
        def leave_scratch():
            (tmp_path / "artifact").touch()
            (tmp_path / "helper.o").write_bytes(b"second file")
            (tmp_path / "dangling").symlink_to(tmp_path / "nowhere")
            # A read-only tree, like the module cache `go build` leaves.
            cache = tmp_path / "cache" / "deep"
            cache.mkdir(parents=True)
            (cache / "pkg").touch()
            cache.chmod(0o500)

        source = self._stage(tmp_path)
        self._build_does(monkeypatch, leave_scratch)

        result = toolchain_grading.build_submission(
            self._submission(), source, tmp_path
        )

        assert result.error is None
        assert sorted(tmp_path.iterdir()) == [result.artifact, source]

    def test_keep_built_patterns_survive_the_clearing(self, tmp_path, monkeypatch):
        def leave_beams():
            (tmp_path / "Elixir.Server.beam").touch()
            (tmp_path / "Elixir.Helper.beam").touch()
            (tmp_path / "scratch.tmp").touch()

        source = self._stage(tmp_path)
        self._build_does(monkeypatch, leave_beams)
        submission = self._submission(
            artifact="Elixir.Server.beam", keep_built=("*.beam",)
        )

        result = toolchain_grading.build_submission(submission, source, tmp_path)

        assert result.error is None
        kept = tmp_path / "Elixir.Helper.beam"
        assert kept.is_file()
        assert not (tmp_path / "scratch.tmp").exists()
        # Kept, but sealed like everything else the run may touch.
        assert kept.stat().st_mode & 0o222 == 0

    def test_a_build_that_rewrote_the_staged_source_is_undone(
        self, tmp_path, monkeypatch
    ):
        def tamper():
            (tmp_path / "artifact").touch()
            (tmp_path / "main.rs").write_bytes(b"not what was submitted")

        source = self._stage(tmp_path)
        self._build_does(monkeypatch, tamper)

        result = toolchain_grading.build_submission(
            self._submission(), source, tmp_path
        )

        assert result.error is None
        assert source.read_bytes() == self.PRISTINE


class TestBuildAsBuilder:
    """The build demotes to the builder uid — the account that can run the
    toolchain — and hands back everything the compilers wrote, which is what
    the build MCP tool shows the student."""

    def _submission(self, commands=1) -> "toolchain_grading.Submission":
        build = tuple(("rustc", "{source}") for _ in range(commands))
        return toolchain_grading.Submission(source="main.rs", build=build)

    def test_the_build_demotes_to_the_builder(self, tmp_path, monkeypatch):
        captured = {}

        def fake_demote(*chown_fds, uid_gid=None, **policy):
            captured["chown_fds"] = chown_fds
            captured["uid_gid"] = uid_gid
            return None

        monkeypatch.setattr(toolchain_grading, "make_demote_fn", fake_demote)
        monkeypatch.setattr(
            toolchain_grading.subprocess,
            "run",
            lambda argv, **kwargs: subprocess.CompletedProcess(
                argv, returncode=1, stderr=b"stopped"
            ),
        )
        source = tmp_path / "main.rs"
        source.write_bytes(b"fn main() {}")

        toolchain_grading.build_submission(self._submission(), source, tmp_path)

        assert captured["uid_gid"] == toolchains.BUILDER_UID
        assert captured["chown_fds"] == (1, 2)

    def test_the_build_gets_a_mount_namespace_of_its_own_in_the_image(
        self, tmp_path, monkeypatch
    ):
        """A compiler that runs student code at compile time must not be able
        to leave anything behind: `-pgmF` and friends are bounded by the
        filesystem the build sees, not by the pinned argv."""
        captured = {}

        def fake_demote(*chown_fds, uid_gid=None, **policy):
            captured.update(policy)
            return None

        monkeypatch.setenv("KAROTTE_CONTAINERIZED", "1")
        monkeypatch.setattr(toolchain_grading, "make_demote_fn", fake_demote)
        monkeypatch.setattr(toolchain_grading, "kill_processes", lambda uid: None)
        monkeypatch.setattr(
            toolchain_grading.subprocess,
            "run",
            lambda argv, **kwargs: subprocess.CompletedProcess(
                argv, returncode=1, stderr=b"stopped"
            ),
        )
        source = tmp_path / "main.rs"
        source.write_bytes(b"fn main() {}")

        toolchain_grading.build_submission(self._submission(), source, tmp_path)

        assert captured["ephemeral_dirs"] == toolchain_grading.EPHEMERAL_BUILD_DIRS
        assert captured["read_only_dirs"] == toolchain_grading.READ_ONLY_BUILD_DIRS

    def test_the_build_cannot_reach_the_platforms_own_mounts_either(
        self, tmp_path, monkeypatch
    ):
        """Covering them only for the run would leave a compile-time hook free
        to copy the interpreter somewhere the run keeps."""
        captured = {}

        def fake_demote(*chown_fds, uid_gid=None, **policy):
            captured.update(policy)
            return None

        monkeypatch.setenv("KAROTTE_CONTAINERIZED", "1")
        monkeypatch.setattr(
            toolchain_grading,
            "platform_tooling_dirs",
            lambda: ("/opt/platform-tooling",),
        )
        monkeypatch.setattr(toolchain_grading, "make_demote_fn", fake_demote)
        monkeypatch.setattr(toolchain_grading, "kill_processes", lambda uid: None)
        monkeypatch.setattr(
            toolchain_grading.subprocess,
            "run",
            lambda argv, **kwargs: subprocess.CompletedProcess(
                argv, returncode=1, stderr=b"stopped"
            ),
        )
        source = tmp_path / "main.rs"
        source.write_bytes(b"fn main() {}")

        toolchain_grading.build_submission(self._submission(), source, tmp_path)

        assert captured["covered_dirs"] == ("/opt/platform-tooling",)

    def test_the_build_cannot_see_the_students_workdir_at_all(
        self, tmp_path, monkeypatch
    ):
        """Read-only is not enough: a source file can name a second file by
        absolute path, and a compiler that reads it is one the single staged
        file no longer bounds. An empty tmpfs is what leaves nothing to name."""
        captured = {}

        def fake_demote(*chown_fds, uid_gid=None, **policy):
            captured.update(policy)
            return None

        monkeypatch.setenv("KAROTTE_CONTAINERIZED", "1")
        monkeypatch.setattr(toolchain_grading, "make_demote_fn", fake_demote)
        monkeypatch.setattr(toolchain_grading, "kill_processes", lambda uid: None)
        monkeypatch.setattr(
            toolchain_grading.subprocess,
            "run",
            lambda argv, **kwargs: subprocess.CompletedProcess(
                argv, returncode=1, stderr=b"stopped"
            ),
        )
        source = tmp_path / "main.rs"
        source.write_bytes(b"fn main() {}")

        toolchain_grading.build_submission(self._submission(), source, tmp_path)

        assert str(STUDENT_WORKDIR) in captured["ephemeral_dirs"]
        assert str(STUDENT_WORKDIR) not in captured["read_only_dirs"]

    def test_the_build_leaves_a_dev_box_mounts_alone(self, tmp_path, monkeypatch):
        captured = {}

        def fake_demote(*chown_fds, uid_gid=None, **policy):
            captured.update(policy)
            return None

        monkeypatch.delenv("KAROTTE_CONTAINERIZED", raising=False)
        monkeypatch.setattr(toolchain_grading, "make_demote_fn", fake_demote)
        monkeypatch.setattr(
            toolchain_grading.subprocess,
            "run",
            lambda argv, **kwargs: subprocess.CompletedProcess(
                argv, returncode=1, stderr=b"stopped"
            ),
        )
        source = tmp_path / "main.rs"
        source.write_bytes(b"fn main() {}")

        toolchain_grading.build_submission(self._submission(), source, tmp_path)

        assert captured == {
            "ephemeral_dirs": (),
            "read_only_dirs": (),
            "covered_dirs": (),
        }

    def test_scratch_stays_inside_the_build_directory(self, tmp_path, monkeypatch):
        """With /tmp fresh and private, a compiler that puts scratch there
        loses it — so TMPDIR has to name somewhere the build owns."""
        captured = {}
        monkeypatch.setattr(
            toolchain_grading.subprocess,
            "run",
            lambda argv, **kwargs: (
                captured.update(kwargs),
                subprocess.CompletedProcess(argv, returncode=1, stderr=b"x"),
            )[1],
        )
        source = tmp_path / "main.rs"
        source.write_bytes(b"fn main() {}")

        toolchain_grading.build_submission(self._submission(), source, tmp_path)

        assert captured["env"]["TMPDIR"] == str(tmp_path)
        assert captured["env"]["HOME"] == str(tmp_path)

    def test_the_result_carries_everything_the_commands_wrote(
        self, tmp_path, monkeypatch
    ):
        outputs = iter([(b"one out\n", b"one err\n"), (b"two out\n", b"two err\n")])

        def run(argv, **kwargs):
            out, err = next(outputs)
            (tmp_path / "artifact").touch()
            return subprocess.CompletedProcess(
                argv, returncode=0, stdout=out, stderr=err
            )

        monkeypatch.setattr(toolchain_grading.subprocess, "run", run)
        source = tmp_path / "main.rs"
        source.write_bytes(b"fn main() {}")

        result = toolchain_grading.build_submission(
            self._submission(commands=2), source, tmp_path
        )

        assert result.error is None
        assert result.stdout == b"one out\ntwo out\n"
        assert result.stderr == b"one err\ntwo err\n"

    def test_a_failing_build_still_hands_its_output_back(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            toolchain_grading.subprocess,
            "run",
            lambda argv, **kwargs: subprocess.CompletedProcess(
                argv, returncode=1, stdout=b"progress", stderr=b"boom"
            ),
        )
        source = tmp_path / "main.rs"
        source.write_bytes(b"fn main() {}")

        result = toolchain_grading.build_submission(
            self._submission(), source, tmp_path
        )

        assert result.error is not None
        assert result.stdout == b"progress"
        assert result.stderr == b"boom"

    def test_the_build_dir_goes_to_the_builder(self, tmp_path, monkeypatch):
        owners = {}
        monkeypatch.setattr(toolchain_grading, "BUILD_ROOT", tmp_path / "build")
        monkeypatch.setattr(
            toolchain_grading,
            "_chown",
            lambda path, uid: owners.__setitem__(path, uid),
        )

        build_dir = toolchain_grading.make_build_dir()

        assert owners[build_dir] == toolchains.BUILDER_UID


class TestArchiveBuildParallelism:
    """The builder installs the enabled languages concurrently. What must stay
    true: every enabled archive lands, one failure neither stops the others nor
    goes unnamed, and each language's output stays in one piece."""

    @pytest.fixture
    def workdir(self, tmp_path, monkeypatch) -> Path:
        monkeypatch.setattr(toolchains, "TOOLCHAINS_DIR", tmp_path / "toolchains")
        return tmp_path / "build"

    @pytest.fixture
    def enabled(self, monkeypatch) -> frozenset[Language]:
        # Mojo carries two archives, so multi-archive ordering is covered too.
        chosen = frozenset({Language.GO, Language.RUST, Language.MOJO})
        monkeypatch.setattr(toolchains, "enabled_languages", lambda: chosen)
        return chosen

    def test_every_enabled_archive_is_installed(self, workdir, monkeypatch, enabled):
        installed = []
        lock = threading.Lock()

        def record(toolchain, archive, workdir):
            with lock:
                installed.append((toolchain.language, archive.urls["x86_64"]))

        monkeypatch.setattr(toolchains, "_install_archive", record)

        toolchains.install_archives(workdir)

        expected = {
            (language, archive.urls["x86_64"])
            for language in enabled
            for archive in TOOLCHAINS[language].archives
        }
        assert set(installed) == expected
        assert len(installed) == len(expected)

    def test_languages_build_concurrently(self, workdir, monkeypatch):
        monkeypatch.setattr(
            toolchains,
            "enabled_languages",
            lambda: frozenset({Language.GO, Language.ZIG}),
        )
        # Each build waits for the other; a serial loop would break the barrier
        # by timeout and fail the install.
        barrier = threading.Barrier(2, timeout=10)
        monkeypatch.setattr(
            toolchains, "_install_archive", lambda t, a, w: barrier.wait()
        )

        toolchains.install_archives(workdir)

        assert not barrier.broken

    def test_one_failure_neither_stops_nor_hides_the_others(
        self, workdir, monkeypatch, enabled
    ):
        installed = []
        lock = threading.Lock()

        def record(toolchain, archive, workdir):
            if toolchain.language is Language.RUST:
                raise RuntimeError("mirror down")
            with lock:
                installed.append(toolchain.language)

        monkeypatch.setattr(toolchains, "_install_archive", record)

        with pytest.raises(SystemExit, match="rust"):
            toolchains.install_archives(workdir)

        assert set(installed) == {Language.GO, Language.MOJO}

    def test_each_language_logs_in_one_piece(self, workdir, monkeypatch, capsys):
        monkeypatch.setattr(
            toolchains,
            "enabled_languages",
            lambda: frozenset({Language.GO, Language.ZIG}),
        )
        barrier = threading.Barrier(2, timeout=10)

        def chatter(toolchain, archive, workdir):
            log = toolchains._build_log_file()
            print(f"{toolchain.language.value} begins", file=log)
            # Both builds are mid-flight when their lines land.
            barrier.wait()
            print(f"{toolchain.language.value} ends", file=log)

        monkeypatch.setattr(toolchains, "_install_archive", chatter)

        toolchains.install_archives(workdir)

        lines = [line for line in capsys.readouterr().err.splitlines() if line]
        for language in ("go", "zig"):
            begins = lines.index(f"{language} begins")
            assert lines[begins + 1] == f"{language} ends"

    def test_run_writes_to_the_build_log_only_when_one_is_set(
        self, tmp_path, monkeypatch
    ):
        captured = {}

        def capture(argv, **kwargs):
            captured.update(kwargs)
            return subprocess.CompletedProcess(argv, returncode=0)

        monkeypatch.setattr(toolchains.subprocess, "run", capture)

        toolchains._run("true")
        assert captured["stdout"] is None and captured["stderr"] is None

        with open(tmp_path / "log", "w") as log:
            toolchains._build_log.file = log
            try:
                toolchains._run("true")
            finally:
                del toolchains._build_log.file
        assert captured["stdout"] is log and captured["stderr"] is log


class TestBuildHalfIsStandaloneStdlib:
    """The Containerfile runs this file as a script under python3.12, on the
    bare image, before the environment package or its venv exist."""

    def test_it_imports_nothing_from_the_environment_package(self):
        import ast

        source = Path(toolchains.__file__).read_text()
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.ImportFrom):
                assert node.module is None or not node.module.startswith("environment")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith("environment")

    def test_it_has_a_phase_for_each_half_of_the_build(self):
        import ast

        source = Path(toolchains.__file__).read_text()
        tree = ast.parse(source)
        main = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "main"
        )
        phases = {
            key.value
            for node in ast.walk(main)
            if isinstance(node, ast.Dict)
            for key in node.keys
            if isinstance(key, ast.Constant)
        }
        assert phases == {"archives", "rpms"}


class TestImageLayers:
    """Each language is copied out of the builder stage on its own, so the
    image carries one layer per language rather than one blob of all of them:
    a registry gzips a layer on a single core, and layers go up in parallel."""

    @pytest.fixture
    def containerfile(self) -> str:
        return (Path(__file__).resolve().parent.parent / "Containerfile").read_text()

    def test_every_language_is_copied_out_on_its_own(self, containerfile: str):
        for language in Language:
            directory = f"{toolchains.TOOLCHAINS_DIR}/{language.value}"
            assert f"COPY --from=toolchains {directory} {directory}\n" in containerfile

    def test_the_whole_tree_is_not_copied_in_one_go(self, containerfile: str):
        bulk = f"COPY --from=toolchains {toolchains.TOOLCHAINS_DIR} "
        assert f"{bulk}{toolchains.TOOLCHAINS_DIR}" not in containerfile

    def test_the_build_leaves_a_directory_for_every_language_to_copy(
        self, tmp_path, monkeypatch
    ):
        """Including the ones this environment disables: the Containerfile
        names all of them, and a COPY with nothing at the path fails the
        build."""
        monkeypatch.setattr(toolchains, "TOOLCHAINS_DIR", tmp_path / "toolchains")
        monkeypatch.setattr(
            toolchains, "enabled_languages", lambda: frozenset({Language.GO})
        )
        monkeypatch.setattr(toolchains, "_install_archive", lambda t, a, w: None)

        toolchains.install_archives(tmp_path / "build")

        for language in Language:
            assert (tmp_path / "toolchains" / language.value).is_dir()


class TestCheckScriptRecipe:
    """`just check-toolchains` builds inside the image the way a run does."""

    @pytest.fixture
    def recipe(self) -> str:
        return (Path(__file__).resolve().parent.parent / "toolchains.just").read_text()

    def test_the_container_can_build_a_mount_namespace(self, recipe: str):
        """`build_submission` gives each build its own mount namespace and
        refuses to build without one, so a container that cannot make one
        fails every check rather than reporting on the sandbox."""
        assert "--cap-add=SYS_ADMIN" in recipe
        assert "--security-opt apparmor=unconfined" in recipe

    def test_languages_are_checked_in_parallel(self, recipe: str):
        """One container per language is independent of the next, and the
        slowest cell alone takes a minute; the loop runs several at once."""
        assert re.search(r"xargs\s.*-P", recipe)

    def test_the_slow_cells_go_first(self, recipe: str):
        """Alphabetical order puts zig, the slowest cell by far, last, so the
        run ends with it alone; the script hands the recipe a better order."""
        assert "check_toolchains.py --list" in recipe


@pytest.fixture
def check_script():
    import importlib.util

    path = Path(__file__).parent.parent / "scripts" / "check_toolchains.py"
    spec = importlib.util.spec_from_file_location("check_toolchains_probe", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestCheckScriptCoverage:
    """Which cells the script checks the scored run of."""

    def test_only_the_python_cell_confines_its_run_from_the_inside(self, check_script):
        """Everything else is confined from outside, and an exemption is the
        one way a cell's scored run goes unchecked."""
        assert check_script.UNCONFINED_CELLS == {Language.PYTHON}

    def test_every_other_cell_has_its_scored_run_checked(self, check_script):
        assert check_script.CONFINED_CELLS == set(Language) - {Language.PYTHON}

    def test_every_cell_has_a_program_to_check_it_with(self, check_script):
        assert set(check_script.PROGRAMS) == set(Language)


class TestTheBuildSandboxVerdict:
    """Reading the probe's report, which is the half of that check with no
    container in it."""

    def report(self, workdir: str, **answers: str) -> str:
        lines = [
            f"{answers.get('planted', 'BLOCKED')} {workdir}/karotte_probe_planted.sh",
            f"{answers.get('read', 'NOREAD')} {workdir}",
            f"{answers.get('write', 'WROTE')} {workdir}",
        ]
        return "\n".join(lines)

    def test_a_build_that_saw_nothing_and_ran_nothing_is_clean(self, check_script):
        workdir = str(check_script.STUDENT_WORKDIR)

        assert check_script.why_the_build_sandbox_leaks(self.report(workdir)) is None

    def test_a_build_that_could_list_the_workdir_is_a_leak(self, check_script):
        workdir = str(check_script.STUDENT_WORKDIR)

        leak = check_script.why_the_build_sandbox_leaks(
            self.report(workdir, read="READ")
        )

        assert leak is not None and "read the student's workdir" in leak

    def test_a_build_that_ran_a_planted_script_is_a_leak(self, check_script):
        workdir = str(check_script.STUDENT_WORKDIR)

        leak = check_script.why_the_build_sandbox_leaks(
            self.report(workdir, planted="RAN")
        )

        assert leak is not None and "planted script" in leak

    def test_a_report_that_never_arrived_is_a_leak(self, check_script):
        """A probe that died before it answered has said nothing, which is not
        the same as having said no."""
        assert check_script.why_the_build_sandbox_leaks("") is not None


class TestCheckScriptBuildEnv:
    """`just check-toolchains` builds as the builder, whose passwd home does
    not exist. A toolchain that wants `~/.cache` fails there and nowhere else,
    so the script's builds carry the same HOME `build_submission` sets."""

    def test_the_builders_commands_carry_a_home_that_exists(
        self, check_script, tmp_path, monkeypatch
    ):
        calls: list[tuple[str, list[str]]] = []

        def record(who: str):
            def run(argv: list[str], cwd: Path) -> subprocess.CompletedProcess:
                calls.append((who, argv))
                return subprocess.CompletedProcess(argv, 0, check_script.OK, "")

            return run

        monkeypatch.setattr(check_script, "_as_builder", record("builder"))
        monkeypatch.setattr(check_script, "_as_student", record("student"))
        monkeypatch.setattr(check_script.shutil, "chown", lambda *args, **kwargs: None)

        check_script.build_one(
            Language.GO,
            tmp_path,
            ("main.go", "package main\n", (("go", "build"),), ("./main",)),
        )

        builder = [argv for who, argv in calls if who == "builder"]
        assert builder and all(
            argv[:2] == ["env", f"HOME={tmp_path}"] for argv in builder
        )


class TestGradingBuildsAreShared:
    """The confined-run, run-flags and escape checks compile the same sources.
    A sealed build directory is root-owned and read-only and a run only
    executes what is in it, so one grading build serves every check."""

    @pytest.fixture
    def builds(self, check_script, tmp_path, monkeypatch) -> list[tuple]:
        import tempfile
        from types import SimpleNamespace

        calls: list[tuple] = []

        def build_submission(submission, staged, build_dir):
            calls.append(
                (submission.source, staged.read_text(), submission.allow_unsafe)
            )
            return SimpleNamespace(error=None)

        monkeypatch.setattr(check_script, "build_submission", build_submission)
        monkeypatch.setattr(
            check_script,
            "make_build_dir",
            lambda: Path(tempfile.mkdtemp(dir=tmp_path)),
        )
        return calls

    program = (
        "main.go",
        "package main\n",
        [["go", "build", "-o", "main", "main.go"]],
        ["./main"],
    )

    def test_the_same_source_is_built_once(self, check_script, builds):
        first, _ = check_script._build_as_grading_would(
            Language.GO, self.program, "package main\n"
        )
        second, _ = check_script._build_as_grading_would(
            Language.GO, self.program, "package main\n"
        )

        assert first == second
        assert len(builds) == 1

    def test_a_different_source_is_its_own_build(self, check_script, builds):
        check_script._build_as_grading_would(
            Language.GO, self.program, "package main\n"
        )
        check_script._build_as_grading_would(
            Language.GO, self.program, "package other\n"
        )

        assert len(builds) == 2

    def test_opening_the_forbid_is_its_own_build(self, check_script, builds):
        """`check_escapes_are_refused` builds an escape twice, once hardened
        and once with `allow_unsafe`, and expects different answers."""
        check_script._build_as_grading_would(
            Language.GO, self.program, "package main\n"
        )
        check_script._build_as_grading_would(
            Language.GO, self.program, "package main\n", allow_unsafe=True
        )

        assert len(builds) == 2


class TestCheckOrder:
    """`just check-toolchains` runs several cells at once; the wall time is
    bounded by the slowest, so those start first."""

    def test_slow_cells_first_then_the_rest_alphabetically(self, check_script):
        order = check_script.check_order(
            {Language.GO, Language.ZIG, Language.ASSEMBLY, Language.KOTLIN}
        )

        assert order == [
            Language.ZIG,
            Language.KOTLIN,
            Language.ASSEMBLY,
            Language.GO,
        ]

    def test_every_language_once(self, check_script):
        order = check_script.check_order(frozenset(Language))

        assert sorted(order) == sorted(Language)
        assert order[: len(check_script.SLOW_CELLS)] == list(check_script.SLOW_CELLS)
