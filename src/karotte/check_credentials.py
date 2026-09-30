"""Reject images that ship credential material.

Secret mounts keep secrets out of layers, but tools persist tokens derived from them
(uv caches index tokens under ``$XDG_DATA_HOME``), so this checks a denylist of known
credential stores after the last build step.
"""

import os
import re
import stat
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path

# Paths whose presence *is* the credential — a token store, a private key, a
# session file. Relative to each home directory; globs are expanded.
_CREDENTIAL_PATHS: tuple[str, ...] = (
    ".local/share/uv/credentials",  # uv's own credential store
    ".config/uv/credentials.toml",
    ".netrc",
    ".git-credentials",
    ".config/gh/hosts.yml",  # GitHub CLI OAuth token
    # Not `.config/gcloud` wholesale: any `gcloud` invocation creates that
    # directory (`configurations/config_default`, `.last_update_check.json`)
    # with no login at all, so an image that merely installs the SDK would fail
    # on config rather than on a credential. Name the files that hold one.
    ".config/gcloud/application_default_credentials.json",
    ".config/gcloud/credentials.db",
    ".config/gcloud/access_tokens.db",
    ".config/gcloud/legacy_credentials",
    ".aws/credentials",
    ".kube/config",
    ".cache/huggingface/token",
    ".huggingface/token",
    # Likewise keyring: `keyringrc.cfg` only names a backend, which uv's
    # `keyring-provider = "subprocess"` needs. These two are the stores that
    # actually hold secrets.
    ".local/share/python_keyring/keyring_pass.cfg",
    ".local/share/python_keyring/crypted_pass.cfg",
    ".config/python_keyring/keyring_pass.cfg",
    ".config/python_keyring/crypted_pass.cfg",
    ".ssh/id_*",
)

# `.ssh/id_*` also matches the public half of a keypair, which is not a secret.
_PUBLIC_KEY_SUFFIXES = (".pub",)

# Config files that are a problem only when they carry a secret inline. A bare
# registry URL or a credential-helper entry is fine and common, so these are
# matched on content rather than existence.
_CREDENTIAL_MARKER_FILES: tuple[str, ...] = (
    ".npmrc",
    ".pypirc",
    ".docker/config.json",
    ".config/pip/pip.conf",
    ".config/uv/uv.toml",
)

_MARKER_KEYS: tuple[str, ...] = (
    "_authtoken",
    "_auth",
    "_password",
    "password",
    "auth",
    "identitytoken",
    "access_token",
    "refresh_token",
    "api_key",
)

# A key only counts when a value actually follows it. `_authToken=${NPM_TOKEN}`
# is the *correct* npm idiom — the point of it is that the token is not in the
# file — and a comment naming `UV_INDEX_PRIVATE_PASSWORD` is prose, not a
# secret. So: reject a line that is a comment (`#`/`;` before the key), and
# require a first value character that is neither end-of-line nor the `$` of an
# interpolation. The optional quotes carry the JSON form, `"auth": "…"`.
_CREDENTIAL_MARKERS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(rf'(?im)^[^#;\n]*\b{key}\b"?\s*[=:]\s*"?(?!\s|$|\$)[^\s"\']')
    for key in _MARKER_KEYS
)

# `https://user:secret@host/...` — a credential smuggled into an index URL.
_URL_USERINFO = re.compile(r"://[^/\s:@]+:[^/\s@]+@")

# Reading a whole file to find a marker is pointless; a credential lives near the
# top of a config file, and this bounds the cost of a pathological one.
_MARKER_READ_BYTES = 64 * 1024


class CredentialInImage(Exception):
    pass


def _is_nonempty_file(path: Path) -> bool:
    """A non-empty regular file, following symlinks.

    `stat()` follows, deliberately: a symlinked `.netrc` pointing at a real
    secret ships that secret, and treating the link as "not a file" would read
    clean. A zero-byte file carries nothing — uv leaves empty `.lock` files
    beside its tokens — and a broken link raises, which is not a credential.
    """
    try:
        info = path.stat()
    except OSError:
        return False
    return stat.S_ISREG(info.st_mode) and info.st_size > 0


def _files_under(path: Path, seen: set[Path]) -> Iterator[Path]:
    """Every non-empty file at `path`, or under it when it is a directory.

    Symlinks are followed, so `seen` (keyed on the resolved path) both stops a
    link that points back up the tree and keeps a file reachable two ways from
    being reported twice.
    """
    try:
        resolved = path.resolve()
    except OSError:
        return
    if resolved in seen:
        return
    seen.add(resolved)

    if path.is_dir():
        for child in sorted(path.iterdir()):
            yield from _files_under(child, seen)
    elif _is_nonempty_file(path):
        yield path


def _has_marker(path: Path) -> bool:
    try:
        with path.open("rb") as f:
            head = f.read(_MARKER_READ_BYTES).decode("utf-8", errors="replace")
    except OSError:
        # Unreadable by root is odd but is not evidence of a credential.
        return False
    return any(m.search(head) for m in _CREDENTIAL_MARKERS) or bool(
        _URL_USERINFO.search(head)
    )


def home_directories(env: Mapping[str, str] | None = None) -> list[Path]:
    """The home directories a build could have written a credential into.

    `/root` is where every `RUN` step lands by default; the student's home and
    any `/home/*` are included because a build step that drops privileges writes
    there instead.
    """
    if env is None:
        env = os.environ

    candidates = [Path("/root")]
    for var in ("HOME", "ROOT_WORKDIR", "STUDENT_WORKDIR", "KAROTTE_WORKDIR"):
        value = env.get(var)
        if value:
            candidates.append(Path(value))
    home_root = Path("/home")
    if home_root.is_dir():
        candidates.extend(sorted(p for p in home_root.iterdir() if p.is_dir()))

    seen: dict[Path, None] = {}
    for path in candidates:
        if path.is_dir():
            seen.setdefault(path.resolve(), None)
    return list(seen)


def relocated_stores(env: Mapping[str, str] | None = None) -> list[Path]:
    """Credential stores that an env var has moved out from under a home directory.

    uv resolves its stores through these in preference to `$HOME`, so a build
    that sets one puts the token somewhere the home-relative list cannot see.
    """
    if env is None:
        env = os.environ

    roots: list[Path] = []
    if value := env.get("UV_CREDENTIALS_DIR"):
        roots.append(Path(value))
    xdg_data_home = env.get("XDG_DATA_HOME")
    if xdg_data_home:
        roots.append(Path(xdg_data_home) / "uv" / "credentials")
    return roots


def find_credentials(
    homes: Sequence[Path] | None = None,
    extra_roots: Sequence[Path] = (),
    env: Mapping[str, str] | None = None,
) -> list[Path]:
    """Every file in the image that looks like shipped credential material."""
    if homes is None:
        homes = home_directories(env)

    found: dict[Path, None] = {}
    seen: set[Path] = set()

    def collect(target: Path) -> None:
        for file in _files_under(target, seen):
            if not file.name.endswith(_PUBLIC_KEY_SUFFIXES):
                found.setdefault(file, None)

    for home in homes:
        for pattern in _CREDENTIAL_PATHS:
            for target in sorted(home.glob(pattern)):
                collect(target)

    for root in extra_roots:
        collect(root)

    for home in homes:
        for relative in _CREDENTIAL_MARKER_FILES:
            path = home / relative
            if _is_nonempty_file(path) and _has_marker(path):
                found.setdefault(path, None)

    return sorted(found)


def check_credentials(
    homes: Sequence[Path] | None = None,
    extra_roots: Sequence[Path] | None = None,
    env: Mapping[str, str] | None = None,
) -> None:
    """Raise if the image ships credential material.

    Raises:
        CredentialInImage: listing every offending path.
    """
    if extra_roots is None:
        extra_roots = relocated_stores(env)

    found = find_credentials(homes, extra_roots, env)
    if not found:
        return

    message = [
        "The image ships credential material. Anyone who can pull the image can read",
        "these files, and a root-only mode does not change that:",
        *(f"  - {p}" for p in found),
        "A build secret mounted with `--mount=type=secret` stays out of the layers, but a",
        "tool handed one may persist what it derives from it, like uv's credential store",
        "under ~/.local/share/uv/credentials.",
        "",
        "Delete the file in the RUN step that created it, and rotate the credential",
        "behind it. It has to be the same step: this check runs at the end and sees only",
        "the final filesystem, so a credential written in one layer and deleted in a",
        "later one passes here and still ships inside the layer that holds it.",
    ]
    raise CredentialInImage("\n".join(message))
