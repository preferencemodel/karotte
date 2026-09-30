"""Tests for scripts/clear_execstack.py, against hand-built ELF images."""

import importlib.util
import struct
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parent.parent / "scripts" / "clear_execstack.py"

spec = importlib.util.spec_from_file_location("clear_execstack", SCRIPT)
assert spec is not None and spec.loader is not None
clear_execstack = importlib.util.module_from_spec(spec)
spec.loader.exec_module(clear_execstack)

RWX = clear_execstack.PF_R | clear_execstack.PF_W | clear_execstack.PF_X
RW = clear_execstack.PF_R | clear_execstack.PF_W

PT_LOAD = 1


def elf(*headers: tuple[int, int]) -> bytes:
    """A minimal 64-bit little-endian ELF holding the given (p_type, p_flags)
    program headers."""
    header_size = clear_execstack.PHDR.size
    image = bytearray(64 + header_size * len(headers))
    image[:4] = clear_execstack.ELF_MAGIC
    image[4] = 2
    image[5] = 1
    struct.pack_into("<Q", image, clear_execstack.PHOFF, 64)
    struct.pack_into("<H", image, clear_execstack.PHENTSIZE, header_size)
    struct.pack_into("<H", image, clear_execstack.PHNUM, len(headers))
    for index, (p_type, p_flags) in enumerate(headers):
        clear_execstack.PHDR.pack_into(
            image, 64 + header_size * index, p_type, p_flags, 0, 0, 0, 0, 0, 0
        )
    return bytes(image)


def headers_of(image: bytes) -> list[tuple[int, int, int]]:
    """(p_type, p_flags, p_align) per program header."""
    offset, size, count = clear_execstack.program_headers(image)
    return [
        (
            *struct.unpack_from("<II", image, offset + size * index),
            struct.unpack_from("<Q", image, offset + size * index + 48)[0],
        )
        for index in range(count)
    ]


def written(tmp_path: Path, image: bytes, mode: int = 0o755) -> Path:
    path = tmp_path / "python3"
    path.write_bytes(image)
    path.chmod(mode)
    return path


class TestClearExecstack:
    def test_an_executable_stack_request_loses_its_execute_bit(self, tmp_path):
        path = written(
            tmp_path, elf((PT_LOAD, RWX), (clear_execstack.PT_GNU_STACK, RWX))
        )

        answer = clear_execstack.clear_execstack(path)

        assert "cleared the executable bit" in answer
        assert headers_of(path.read_bytes()) == [
            (PT_LOAD, RWX, 0),
            (clear_execstack.PT_GNU_STACK, RW, 0),
        ]

    def test_a_non_executable_request_is_left_alone(self, tmp_path):
        image = elf((clear_execstack.PT_GNU_STACK, RW))
        path = written(tmp_path, image)

        answer = clear_execstack.clear_execstack(path)

        assert "already" in answer
        assert path.read_bytes() == image

    def test_a_missing_header_is_conjured_from_a_note(self, tmp_path):
        path = written(
            tmp_path,
            elf((PT_LOAD, RWX), (clear_execstack.PT_NOTE, clear_execstack.PF_R)),
        )

        answer = clear_execstack.clear_execstack(path)

        assert "PT_NOTE" in answer
        assert headers_of(path.read_bytes()) == [
            (PT_LOAD, RWX, 0),
            (clear_execstack.PT_GNU_STACK, RW, 0x10),
        ]

    def test_nothing_to_fix_and_nothing_to_spend_refuses(self, tmp_path):
        path = written(tmp_path, elf((PT_LOAD, RWX)))

        with pytest.raises(SystemExit, match="no PT_GNU_STACK"):
            clear_execstack.clear_execstack(path)

    def test_something_that_is_not_an_elf_refuses(self, tmp_path):
        path = written(tmp_path, b"#!/bin/sh\n" + bytes(64))

        with pytest.raises(SystemExit, match="ELF"):
            clear_execstack.clear_execstack(path)

    def test_the_mode_survives_the_rewrite(self, tmp_path):
        path = written(tmp_path, elf((clear_execstack.PT_GNU_STACK, RWX)), mode=0o555)

        clear_execstack.clear_execstack(path)

        assert path.stat().st_mode & 0o7777 == 0o555
