"""Tell the loader an ELF executable does not want an executable stack. uv's
CPython carries no PT_GNU_STACK header at all, which glibc reads as a request
for one: every thread it starts gets an 8MB rwx stack."""

import os
import struct
import sys
from pathlib import Path

PT_NOTE = 4
PT_GNU_STACK = 0x6474E551

PF_R = 0x4
PF_W = 0x2
PF_X = 0x1

ELF_MAGIC = b"\x7fELF"

PHOFF = 0x20
PHENTSIZE = 0x36
PHNUM = 0x38

PHDR = struct.Struct("<IIQQQQQQ")


def program_headers(image: bytes) -> tuple[int, int, int]:
    if image[:4] != ELF_MAGIC or image[4] != 2 or image[5] != 1:
        raise SystemExit("only 64-bit little-endian ELF is understood")
    (offset,) = struct.unpack_from("<Q", image, PHOFF)
    (size,) = struct.unpack_from("<H", image, PHENTSIZE)
    (count,) = struct.unpack_from("<H", image, PHNUM)
    if size != PHDR.size:
        raise SystemExit(f"program headers are {size} bytes, not {PHDR.size}")
    return offset, size, count


def clear_execstack(path: Path) -> str:
    image = bytearray(path.read_bytes())
    offset, size, count = program_headers(image)

    marked = [
        at
        for index in range(count)
        if (at := offset + index * size)
        and struct.unpack_from("<I", image, at)[0] == PT_GNU_STACK
    ]
    if marked:
        (at,) = marked
        (flags,) = struct.unpack_from("<I", image, at + 4)
        if not flags & PF_X:
            return "already asks for a non-executable stack"
        struct.pack_into("<I", image, at + 4, flags & ~PF_X)
        replace(path, image)
        return "cleared the executable bit on its PT_GNU_STACK"

    notes = [
        at
        for index in range(count)
        if (at := offset + index * size)
        and struct.unpack_from("<I", image, at)[0] == PT_NOTE
    ]
    if not notes:
        raise SystemExit(f"{path}: no PT_GNU_STACK to fix and no PT_NOTE to spend")

    PHDR.pack_into(image, notes[0], PT_GNU_STACK, PF_R | PF_W, 0, 0, 0, 0, 0, 0x10)
    replace(path, image)
    return "turned a PT_NOTE into a PT_GNU_STACK asking for a writable stack"


def replace(path: Path, image: bytes) -> None:
    """Write beside the original and rename over it. Opening a running
    executable for writing is ETXTBSY, and this may well be the interpreter
    running this script."""
    written = path.with_name(f"{path.name}.execstack")
    written.write_bytes(image)
    written.chmod(path.stat().st_mode & 0o7777)
    os.replace(written, path)


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(f"usage: {sys.argv[0]} <executable>...")
    for name in sys.argv[1:]:
        path = Path(name).resolve()
        print(f"{path}: {clear_execstack(path)}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
