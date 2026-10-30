import os
import stat
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import BaseModel

from karotte.protected_store import (
    PermissionCheckError,
    ProtectedStore,
    SecretNotFoundError,
)


class DummyModel(BaseModel):
    name: str = "default"
    value: int = 0


class NestedModel(BaseModel):
    inner: DummyModel = DummyModel()
    flag: bool = False


class TestWrite:
    def test_write_string(self, tmp_path: Path):
        store = ProtectedStore(directory=tmp_path)

        store.write("token", "sk-abc123")

        assert (tmp_path / "token").read_text() == "sk-abc123"

    def test_write_bytes(self, tmp_path: Path):
        store = ProtectedStore(directory=tmp_path)
        data = b"\x00\x01\x02\xff"

        store.write("binary_secret", data)

        assert (tmp_path / "binary_secret").read_bytes() == data

    def test_write_pydantic_model(self, tmp_path: Path):
        store = ProtectedStore(directory=tmp_path)

        store.write("config", DummyModel(name="test", value=42))

        raw = (tmp_path / "config").read_text()
        loaded = DummyModel.model_validate_json(raw)
        assert loaded.name == "test"
        assert loaded.value == 42

    def test_write_nested_pydantic_model(self, tmp_path: Path):
        store = ProtectedStore(directory=tmp_path)
        model = NestedModel(inner=DummyModel(name="nested", value=99), flag=True)

        store.write("nested", model)

        raw = (tmp_path / "nested").read_text()
        loaded = NestedModel.model_validate_json(raw)
        assert loaded.inner.name == "nested"
        assert loaded.inner.value == 99
        assert loaded.flag is True

    def test_write_creates_directory_if_missing(self, tmp_path: Path):
        nested = tmp_path / "a" / "b" / "c"
        store = ProtectedStore(directory=nested)

        store.write("secret", "value")

        assert nested.is_dir()
        assert (nested / "secret").read_text() == "value"

    def test_write_sets_file_permissions_to_0600(self, tmp_path: Path):
        store = ProtectedStore(directory=tmp_path)

        store.write("secret", "value")

        mode = (tmp_path / "secret").stat().st_mode
        assert stat.S_IMODE(mode) == stat.S_IRUSR | stat.S_IWUSR

    def test_write_overwrites_existing(self, tmp_path: Path):
        store = ProtectedStore(directory=tmp_path)
        store.write("key", "first")

        store.write("key", "second")

        assert (tmp_path / "key").read_text() == "second"

    def test_write_multiple_secrets(self, tmp_path: Path):
        store = ProtectedStore(directory=tmp_path)

        store.write("a", "alpha")
        store.write("b", "bravo")
        store.write("c", DummyModel(name="charlie"))

        assert (tmp_path / "a").read_text() == "alpha"
        assert (tmp_path / "b").read_text() == "bravo"
        assert (
            DummyModel.model_validate_json((tmp_path / "c").read_text()).name
            == "charlie"
        )

    def test_write_empty_string(self, tmp_path: Path):
        store = ProtectedStore(directory=tmp_path)

        store.write("empty", "")

        assert (tmp_path / "empty").read_text() == ""

    def test_write_empty_bytes(self, tmp_path: Path):
        store = ProtectedStore(directory=tmp_path)

        store.write("empty", b"")

        assert (tmp_path / "empty").read_bytes() == b""

    def test_write_sets_directory_permissions_to_0700(self, tmp_path: Path):
        store_dir = tmp_path / "store"
        store = ProtectedStore(directory=store_dir)

        store.write("secret", "value")

        mode = store_dir.stat().st_mode
        assert stat.S_IMODE(mode) == stat.S_IRWXU

    def test_write_rejects_path_traversal(self, tmp_path: Path):
        store = ProtectedStore(directory=tmp_path)

        with pytest.raises(ValueError, match="Invalid secret name"):
            store.write("../escape", "bad")

    def test_write_rejects_nested_path(self, tmp_path: Path):
        store = ProtectedStore(directory=tmp_path)

        with pytest.raises(ValueError, match="Invalid secret name"):
            store.write("sub/dir", "bad")

    def test_write_rejects_empty_name(self, tmp_path: Path):
        store = ProtectedStore(directory=tmp_path)

        with pytest.raises(ValueError, match="Invalid secret name"):
            store.write("", "bad")

    def test_write_rejects_whitespace_only_name(self, tmp_path: Path):
        store = ProtectedStore(directory=tmp_path)

        with pytest.raises(ValueError, match="Invalid secret name"):
            store.write("   ", "bad")


class TestRead:
    def test_read_string(self, tmp_path: Path):
        store = ProtectedStore(directory=tmp_path)
        store.write("token", "sk-abc123")

        result = store.read("token", str)

        assert result == "sk-abc123"

    def test_read_bytes(self, tmp_path: Path):
        store = ProtectedStore(directory=tmp_path)
        data = b"\x00\x01\x02\xff"
        store.write("binary", data)

        result = store.read("binary", bytes)

        assert result == data

    def test_read_pydantic_model(self, tmp_path: Path):
        store = ProtectedStore(directory=tmp_path)
        store.write("config", DummyModel(name="test", value=42))

        result = store.read("config", DummyModel)

        assert result.name == "test"
        assert result.value == 42

    def test_read_nested_pydantic_model(self, tmp_path: Path):
        store = ProtectedStore(directory=tmp_path)
        model = NestedModel(inner=DummyModel(name="nested", value=99), flag=True)
        store.write("nested", model)

        result = store.read("nested", NestedModel)

        assert result.inner.name == "nested"
        assert result.inner.value == 99
        assert result.flag is True

    def test_read_nonexistent_raises_secret_not_found_error(self, tmp_path: Path):
        store = ProtectedStore(directory=tmp_path)

        with pytest.raises(SecretNotFoundError, match="missing"):
            store.read("missing", str)

    def test_read_from_nonexistent_directory_raises_secret_not_found_error(
        self, tmp_path: Path
    ):
        store = ProtectedStore(directory=tmp_path / "nonexistent")

        with pytest.raises(SecretNotFoundError):
            store.read("anything", str)

    def test_read_invalid_json_as_model_raises(self, tmp_path: Path):
        store = ProtectedStore(directory=tmp_path)
        store.write("bad", "not valid json")

        with pytest.raises(Exception):
            store.read("bad", DummyModel)

    def test_read_rejects_path_traversal(self, tmp_path: Path):
        store = ProtectedStore(directory=tmp_path)

        with pytest.raises(ValueError, match="Invalid secret name"):
            store.read("../escape", str)

    def test_read_rejects_empty_name(self, tmp_path: Path):
        store = ProtectedStore(directory=tmp_path)

        with pytest.raises(ValueError, match="Invalid secret name"):
            store.read("", str)


class TestClear:
    def test_clear_removes_all_files(self, tmp_path: Path):
        store = ProtectedStore(directory=tmp_path)
        store.write("a", "alpha")
        store.write("b", "bravo")
        store.write("c", b"bytes")

        store.clear()

        assert list(tmp_path.iterdir()) == []

    def test_clear_handles_empty_directory(self, tmp_path: Path):
        store = ProtectedStore(directory=tmp_path)

        store.clear()  # Should not raise

    def test_clear_handles_nonexistent_directory(self, tmp_path: Path):
        store = ProtectedStore(directory=tmp_path / "nonexistent")

        store.clear()  # Should not raise


class TestRoundtrip:
    """Verify write then read produces the original data."""

    def test_string_roundtrip(self, tmp_path: Path):
        store = ProtectedStore(directory=tmp_path)
        store.write("s", "hello world")
        assert store.read("s", str) == "hello world"

    def test_bytes_roundtrip(self, tmp_path: Path):
        store = ProtectedStore(directory=tmp_path)
        data = bytes(range(256))
        store.write("b", data)
        assert store.read("b", bytes) == data

    def test_model_roundtrip(self, tmp_path: Path):
        store = ProtectedStore(directory=tmp_path)
        original = DummyModel(name="roundtrip", value=999)
        store.write("m", original)
        assert store.read("m", DummyModel) == original

    def test_model_with_defaults_roundtrip(self, tmp_path: Path):
        store = ProtectedStore(directory=tmp_path)
        original = DummyModel()
        store.write("defaults", original)
        assert store.read("defaults", DummyModel) == original

    def test_unicode_string_roundtrip(self, tmp_path: Path):
        store = ProtectedStore(directory=tmp_path)
        text = "こんにちは 🌍 café"
        store.write("unicode", text)
        assert store.read("unicode", str) == text

    def test_multiline_string_roundtrip(self, tmp_path: Path):
        store = ProtectedStore(directory=tmp_path)
        text = "line1\nline2\nline3\n"
        store.write("multi", text)
        assert store.read("multi", str) == text


class TestSeparateInstances:
    """Verify that separate ProtectedStore instances sharing a directory work correctly."""

    def test_write_from_one_read_from_another(self, tmp_path: Path):
        writer = ProtectedStore(directory=tmp_path)
        reader = ProtectedStore(directory=tmp_path)

        writer.write("shared", DummyModel(name="shared", value=7))

        result = reader.read("shared", DummyModel)
        assert result.name == "shared"
        assert result.value == 7

    def test_clear_from_one_affects_another(self, tmp_path: Path):
        store1 = ProtectedStore(directory=tmp_path)
        store2 = ProtectedStore(directory=tmp_path)

        store1.write("secret", "value")
        store2.clear()

        with pytest.raises(SecretNotFoundError):
            store1.read("secret", str)

    def test_different_directories_are_isolated(self, tmp_path: Path):
        store1 = ProtectedStore(directory=tmp_path / "dir1")
        store2 = ProtectedStore(directory=tmp_path / "dir2")

        store1.write("key", "from_store1")
        store2.write("key", "from_store2")

        assert store1.read("key", str) == "from_store1"
        assert store2.read("key", str) == "from_store2"


def _fake_run(getent_stdout: str = "student:x:1000:1000::/home/student:/bin/sh\n"):
    """Build a subprocess.run stand-in that dispatches on the command.

    ``getent`` resolves the uid to ``getent_stdout`` (empty string means
    the uid is unknown); ``runuser`` returns ``runuser_rc``, captured
    per-call so assertions can inspect the exact argv used.
    """
    calls: list[list[str]] = []

    def make(runuser_rc: int):
        def run(argv: list[str], *_args: object, **_kwargs: object):
            calls.append(argv)
            # argv[0] is an absolute path (trusted_binary), so dispatch on the
            # basename rather than the bare command name.
            match os.path.basename(argv[0]):
                case "getent":
                    rc = 0 if getent_stdout else 2
                    return subprocess.CompletedProcess(argv, rc, stdout=getent_stdout)
                case "runuser":
                    return subprocess.CompletedProcess(argv, runuser_rc)
                case _:
                    raise AssertionError(f"unexpected command: {argv}")

        return run, calls

    return make


class TestCheckPermissions:
    def test_skips_when_karotte_demote_id_not_set(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.delenv("KAROTTE_DEMOTE_ID", raising=False)
        store = ProtectedStore(directory=tmp_path)

        store.check_permissions()  # Should not raise

    def test_runuser_invoked_with_resolved_username_not_hash_uid(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """runuser must get the real username; the ``#<uid>`` sudo-ism
        makes runuser error out (non-zero) regardless of access, which
        would make the check silently fail-open."""
        monkeypatch.setenv("KAROTTE_DEMOTE_ID", "1000")
        store = ProtectedStore(directory=tmp_path)

        run, calls = _fake_run()(runuser_rc=1)
        with patch("karotte.protected_store.subprocess.run", side_effect=run):
            store.check_permissions()

        runuser_call = next(c for c in calls if os.path.basename(c[0]) == "runuser")
        assert "student" in runuser_call
        assert "#1000" not in runuser_call

    def test_passes_when_student_cannot_read(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("KAROTTE_DEMOTE_ID", "1000")
        store = ProtectedStore(directory=tmp_path)

        run, _ = _fake_run()(runuser_rc=1)
        with patch("karotte.protected_store.subprocess.run", side_effect=run):
            store.check_permissions()  # Should not raise

    def test_raises_when_student_can_read(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("KAROTTE_DEMOTE_ID", "1000")
        store = ProtectedStore(directory=tmp_path)

        run, _ = _fake_run()(runuser_rc=0)
        with patch("karotte.protected_store.subprocess.run", side_effect=run):
            with pytest.raises(PermissionCheckError, match="uid=1000"):
                store.check_permissions()

    def test_raises_when_uid_cannot_be_resolved(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """If the uid has no passwd entry we cannot verify anything, so
        fail closed rather than skipping the check."""
        monkeypatch.setenv("KAROTTE_DEMOTE_ID", "1000")
        store = ProtectedStore(directory=tmp_path)

        run, _ = _fake_run(getent_stdout="")(runuser_rc=1)
        with patch("karotte.protected_store.subprocess.run", side_effect=run):
            with pytest.raises(PermissionCheckError, match="resolve"):
                store.check_permissions()

    def test_cleans_up_canary_on_success(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("KAROTTE_DEMOTE_ID", "1000")
        store = ProtectedStore(directory=tmp_path)

        run, _ = _fake_run()(runuser_rc=1)
        with patch("karotte.protected_store.subprocess.run", side_effect=run):
            store.check_permissions()

        assert not (tmp_path / ".permission_check").exists()

    def test_cleans_up_canary_on_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("KAROTTE_DEMOTE_ID", "1000")
        store = ProtectedStore(directory=tmp_path)

        run, _ = _fake_run()(runuser_rc=0)
        with patch("karotte.protected_store.subprocess.run", side_effect=run):
            with pytest.raises(PermissionCheckError):
                store.check_permissions()

        assert not (tmp_path / ".permission_check").exists()
