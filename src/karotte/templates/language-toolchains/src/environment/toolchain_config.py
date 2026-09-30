"""Which language toolchains this environment's image carries.

This file is the environment's to edit; the machinery in `toolchains.py` is the
template's. Add `Language` values from `environment.toolchains` as plain
strings (e.g. "python", "rust", "c_cpp") and rebuild the image. Empty means the
sealing machinery is in place but no toolchain is built in.

Every enabled language grows the image and the build: its archives are
downloaded and unpacked at build time (hundreds of MB to a few GB each), and
"erlang_elixir" compiles OTP from source.
"""

ENABLED_LANGUAGES: frozenset[str] = frozenset()
