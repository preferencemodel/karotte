"""What the grader needs to know about a compiled artifact before launching
one: whether the loader will apply the run's preloaded confinement, and whether
the artifact carries a hook that runs in front of it."""

import struct
from pathlib import Path

ELF_MAGIC = b"\x7fELF"
ELFCLASS64 = 2
ELFDATA2LSB = 1

EM_X86_64 = 62
EM_AARCH64 = 183

MACHINES = {EM_X86_64: "x86_64", EM_AARCH64: "aarch64"}

LOADER = {
    "x86_64": "/lib64/ld-linux-x86-64.so.2",
    "aarch64": "/lib/ld-linux-aarch64.so.1",
}

PT_LOAD = 1
PT_DYNAMIC = 2
PT_INTERP = 3

DT_NULL = 0
DT_PLTRELSZ = 2
DT_RELA = 7
DT_RELASZ = 8
DT_RELAENT = 9
DT_PLTREL = 20
DT_JMPREL = 23
DT_PREINIT_ARRAY = 32

R_IRELATIVE = {"x86_64": 37, "aarch64": 1032}

MACHINE = 0x12
PHOFF = 0x20
PHENTSIZE = 0x36
PHNUM = 0x38

PHDR = struct.Struct("<IIQQQQQQ")
DYN = struct.Struct("<qQ")
RELA = struct.Struct("<QQq")

Header = tuple[int, int, int, int]


def program_headers(image: bytes) -> list[Header]:
    """Every program header, as (type, offset in the file, size in the file,
    virtual address)."""
    if image[:4] != ELF_MAGIC:
        raise ValueError("not an ELF file")
    if image[4] != ELFCLASS64 or image[5] != ELFDATA2LSB:
        raise ValueError("only 64-bit little-endian ELF is understood")
    (offset,) = struct.unpack_from("<Q", image, PHOFF)
    (size,) = struct.unpack_from("<H", image, PHENTSIZE)
    (count,) = struct.unpack_from("<H", image, PHNUM)
    if size != PHDR.size:
        raise ValueError(f"program headers are {size} bytes, not {PHDR.size}")

    headers = []
    for index in range(count):
        kind, _flags, at, vaddr, _paddr, filesz, _memsz, _align = PHDR.unpack_from(
            image, offset + index * size
        )
        headers.append((kind, at, filesz, vaddr))
    return headers


def dynamic_tags(image: bytes, headers: list[Header]) -> dict[int, int]:
    tags = {}
    for kind, at, size, _vaddr in headers:
        if kind != PT_DYNAMIC:
            continue
        for step in range(0, size, DYN.size):
            tag, value = DYN.unpack_from(image, at + step)
            if tag == DT_NULL:
                break
            tags[tag] = value
    return tags


def file_offset(headers: list[Header], vaddr: int) -> int:
    """Where a virtual address sits in the file. Raises when no segment covers
    it, which is a file the loader could not have relocated either."""
    for kind, at, filesz, start in headers:
        if kind == PT_LOAD and start <= vaddr < start + filesz:
            return at + (vaddr - start)
    raise ValueError(f"no PT_LOAD segment holds address {vaddr:#x}")


def machine(image: bytes) -> str:
    """The architecture this artifact is for, as `uname -m` spells it."""
    (value,) = struct.unpack_from("<H", image, MACHINE)
    if value not in MACHINES:
        raise ValueError(f"unknown ELF machine {value}")
    return MACHINES[value]


def interpreter(image: bytes, headers: list[Header]) -> str | None:
    """The program the kernel would start this artifact with, or None where it
    names none and there is no loader at all.

    The whole segment, not the first string in it: a submission's own `.interp`
    section is linked in beside the one the loader path produced, and only the
    first of the two is what runs.
    """
    for kind, at, size, _vaddr in headers:
        if kind != PT_INTERP:
            continue
        name = image[at : at + size]
        if not name.endswith(b"\0"):
            raise ValueError("PT_INTERP is not a terminated string")
        return name[:-1].decode(errors="replace")
    return None


def ifunc_relocations(image: bytes, headers: list[Header], tags: dict[int, int]) -> int:
    """How many relocations name a resolver for the loader to call."""
    name = machine(image)

    tables = [(DT_RELA, DT_RELASZ)]
    if tags.get(DT_PLTREL) == DT_RELA:
        tables.append((DT_JMPREL, DT_PLTRELSZ))

    found = 0
    for at_tag, size_tag in tables:
        if at_tag not in tags or size_tag not in tags:
            continue
        at = file_offset(headers, tags[at_tag])
        size = tags[size_tag]
        if size:
            file_offset(headers, tags[at_tag] + size - 1)
        for step in range(0, size - RELA.size + 1, RELA.size):
            _where, info, _addend = RELA.unpack_from(image, at + step)
            if info & 0xFFFFFFFF == R_IRELATIVE[name]:
                found += 1
    return found


def why_the_confinement_would_not_hold(path: Path) -> str | None:
    """Why this artifact would run outside the preloaded confinement, or None:
    it is statically linked, it asks for a loader that is not the image's, or it
    carries a .preinit_array hook or ifunc resolver the loader runs before the
    confinement's constructor."""
    try:
        image = path.read_bytes()
        headers = program_headers(image)
        tags = dynamic_tags(image, headers)
        loader = LOADER[machine(image)]
        interp = interpreter(image, headers)
        resolvers = ifunc_relocations(image, headers, tags)
    except (OSError, ValueError, KeyError, struct.error, IndexError) as e:
        return f"could not be read as an ELF executable: {e}"

    if interp is None:
        return (
            "is statically linked (no PT_INTERP), so the loader would not read "
            "LD_PRELOAD and the run would be unconfined"
        )
    if interp != loader:
        return (
            f"asks for {interp!r} as its interpreter rather than {loader}, so "
            "the kernel would start a program of the submission's choosing "
            "instead of the loader that reads LD_PRELOAD"
        )
    if DT_PREINIT_ARRAY in tags:
        return (
            "carries a .preinit_array hook, which the loader runs before the "
            "confinement's constructor"
        )
    if resolvers:
        return (
            f"carries {resolvers} ifunc resolver(s), which the loader runs while "
            "relocating, before the confinement's constructor"
        )
    return None
