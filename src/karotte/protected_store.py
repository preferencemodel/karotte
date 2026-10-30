import os
import stat
import subprocess
from pathlib import Path
from typing import Final, overload

from pydantic import BaseModel

from karotte.subprocess import trusted_binary

PROTECTED_STORE_DIR: Final = Path("~/.config/karotte/protected").expanduser()


class SecretNotFoundError(Exception):
    def __init__(self, name: str, directory: Path) -> None:
        super().__init__(f"Secret {name!r} not found in {directory}")


class PermissionCheckError(Exception):
    pass


def _resolve_username(uid: int) -> str:
    """Resolve a numeric uid to its login name via ``getent passwd``.

    ``runuser -u`` requires a real user name. Unlike ``sudo`` it does not
    accept the ``#<uid>`` form: passing it makes runuser exit non-zero
    with a "user does not exist" error that is indistinguishable from a
    genuine permission denial, which would make the permission check
    silently fail-open.

    Raises:
        PermissionCheckError: If the uid has no passwd entry.
    """
    result = subprocess.run(
        [trusted_binary("getent"), "passwd", str(uid)],
        capture_output=True,
        text=True,
        timeout=5,
    )
    if result.returncode != 0 or not result.stdout.strip():
        raise PermissionCheckError(f"Could not resolve uid={uid} to a username")
    return result.stdout.split(":", 1)[0]


class ProtectedStore:
    """A file-based store for sharing data across processes.

    Files are written with permissions so that only the owning user
    (root) can read them. This prevents the student from accessing the data.

    Use this to share configuration or secrets between the main process,
    MCP server, judges, and scoring scripts — without leaking to the student.

    Example::

        from pydantic import BaseModel
        from karotte.protected_store import ProtectedStore

        class MyEnvConfig(BaseModel):
            answer_seed: int = 0
            max_steps: int = 10

        # Write (e.g., in Task.pre_hook):
        ProtectedStore().write("my_env_config", MyEnvConfig(answer_seed=42))

        # Read (e.g., in a scoring script or judge):
        config = ProtectedStore().read("my_env_config", MyEnvConfig)

        # Also works with plain strings or bytes:
        store.write("api_token", "sk-abc123")
        token = store.read("api_token", str)
    """

    def __init__(self, directory: Path = PROTECTED_STORE_DIR) -> None:
        self._directory: Final = directory

    def _resolve_path(self, name: str) -> Path:
        """Resolve and validate a secret name to a path within the store directory.

        Raises:
            ValueError: If the name resolves outside the store directory.
        """
        path = (self._directory / name).resolve()
        if not name or not name.strip() or path.parent != self._directory.resolve():
            raise ValueError(f"Invalid secret name: {name!r}")
        return path

    def write(self, name: str, data: str | bytes | BaseModel) -> None:
        """Write a secret to the store.

        Args:
            name: Identifier for the secret. Used as the filename.
            data: The data to store. Pydantic models are serialized as JSON.
        """
        self._directory.mkdir(parents=True, exist_ok=True)
        self._directory.chmod(stat.S_IRWXU)
        path = self._resolve_path(name)

        # Owner-only read/write. Using os.open (not Path.write_*) so the
        # file is created with restrictive permissions from the start,
        # avoiding a TOCTOU window where it would be world-readable.
        mode = stat.S_IRUSR | stat.S_IWUSR

        if isinstance(data, BaseModel):
            content = data.model_dump_json().encode()
        elif isinstance(data, str):
            content = data.encode()
        else:
            content = data

        # Write to a temp file then atomically replace, so readers never
        # see a partially-written file.
        tmp = f"{path}.tmp"
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, mode)
        with os.fdopen(fd, "wb") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())

        os.replace(tmp, path)

    @overload
    def read(self, name: str, type: type[str]) -> str: ...
    @overload
    def read(self, name: str, type: type[bytes]) -> bytes: ...
    @overload
    def read[T: BaseModel](self, name: str, type: type[T]) -> T: ...

    def read(self, name: str, type: type) -> str | bytes | BaseModel:
        """Read a secret from the store.

        Args:
            name: Identifier for the secret. Must match the name used in write().
            type: The type to return. Pass ``str`` for text, ``bytes`` for raw
                bytes, or a Pydantic model class to deserialize from JSON.

        Raises:
            SecretNotFoundError: If no secret with the given name exists.
        """
        path = self._resolve_path(name)
        if not path.is_file():
            raise SecretNotFoundError(name, self._directory)

        if type is str:
            return path.read_text()
        if type is bytes:
            return path.read_bytes()
        return type.model_validate_json(path.read_text())

    def clear(self) -> None:
        """Remove all secrets from the store."""
        if not self._directory.is_dir():
            return
        for f in self._directory.iterdir():
            f.unlink()

    def check_permissions(self) -> None:
        """Verify that the student user cannot access secrets in the store.

        Writes a canary secret, attempts to read it as the student user,
        and asserts the read fails. Cleans up the canary afterwards.

        Only runs inside containers (requires ``KAROTTE_DEMOTE_ID`` to be set
        and ``runuser`` to be available). Skips silently otherwise.

        Raises:
            PermissionCheckError: If the student user can read the canary secret.
        """
        if "KAROTTE_DEMOTE_ID" not in os.environ:
            return

        canary_name = ".permission_check"
        self.write(canary_name, "canary")

        try:
            uid = int(os.environ["KAROTTE_DEMOTE_ID"])
            username = _resolve_username(uid)
            result = subprocess.run(
                [
                    trusted_binary("runuser"),
                    "-u",
                    username,
                    "--",
                    "cat",
                    str(self._directory / canary_name),
                ],
                capture_output=True,
                timeout=5,
            )
            if result.returncode == 0:
                raise PermissionCheckError(
                    f"Student user (uid={uid}) was able to read {self._directory / canary_name}"
                )
        finally:
            (self._directory / canary_name).unlink(missing_ok=True)
