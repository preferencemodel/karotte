"""Resolve privileged binaries without consulting the (student-writable) PATH."""

import shutil
from typing import Final

TRUSTED_BIN_PATH: Final = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
"""Search path for privileged binaries, holding only root-owned system dirs.

The student's bash session and CLI agents put the student-writable
:data:`STUDENT_VENV` ``/bin`` first on their PATH, and an environment image may
still put it on the image-wide PATH too. Resolving a security-critical binary by
bare name against such a PATH would exec a student-planted file as root.
Resolving against this fixed path never consults a student-writable directory.
"""

STUDENT_VENV: Final = "/workdir/.venv"
"""The student's venv. Put on PATH only for the student's own processes (the bash
tool's ``python_venv``, CLI agents), never image-wide: there it would lead root's
PATH too."""


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
