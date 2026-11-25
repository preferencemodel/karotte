"""Resolve privileged binaries without consulting the (student-writable) PATH."""

import shutil
from typing import Final

TRUSTED_BIN_PATH: Final = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
"""Search path for privileged binaries, holding only root-owned system dirs.

The server runs as root with a PATH that leads with the student-writable
``/workdir/.venv/bin`` — the bash session prepends it (see the ``python_venv``
note in :mod:`karotte.tools.bash`) and the image ``ENV`` sets it too. Resolving a
security-critical binary by bare name against that PATH would exec a
student-planted file as root. Resolving against this fixed path never consults a
student-writable directory.
"""


def trusted_binary(name: str) -> str:
    """Absolute path to a system binary, resolved without consulting ``$PATH``.

    Looks ``name`` up only in the root-owned directories of
    :data:`TRUSTED_BIN_PATH`, so a privileged caller can exec it without a
    student-writable PATH entry shadowing it. Raises ``FileNotFoundError`` if the
    binary is absent there, rather than falling back to the untrusted PATH.
    """
    resolved = shutil.which(name, path=TRUSTED_BIN_PATH)
    if resolved is None:
        raise FileNotFoundError(
            f"Required system binary {name!r} not found on the trusted search path {TRUSTED_BIN_PATH!r}"
        )
    return resolved
