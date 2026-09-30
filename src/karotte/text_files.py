import codecs
from pathlib import Path

NOT_TEXT = "Not a UTF-8 text file: {}. The file tools only handle text; use `bash` for binary files."


def decode_text(data: bytes, file_path: str | Path, *, complete: bool = True) -> str:
    """Decode a file the tools treat as text; ``complete=False`` forgives a cut-off last character."""
    try:
        return codecs.getincrementaldecoder("utf-8")().decode(data, final=complete)
    except UnicodeDecodeError:
        raise ValueError(NOT_TEXT.format(file_path)) from None
