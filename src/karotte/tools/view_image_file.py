import asyncio
import base64
import io
import os
from pathlib import Path
from typing import final

from fastmcp.tools.tool import ToolResult
from fastmcp.utilities.types import Image as FastMCPImage
from karotte import ToolBase, demoted
from karotte.demoted import drain_bounded as _drain_bounded
from karotte.demoted import open_error, open_regular_as_stdin
from karotte.subprocess import make_demote_fn
from pydantic import BaseModel

BASE64_PATH = "/usr/bin/base64"

_MAX_STDERR_BYTES = 16 * 1024


def _b64_encoded_len(n_bytes: int) -> int:
    return ((n_bytes + 2) // 3) * 4


async def _read_capped(stream: asyncio.StreamReader, cap: int) -> tuple[bytes, bool]:
    """Read until EOF or just past ``cap``; returns (bytes, hit_cap)."""
    buf = bytearray()
    while len(buf) <= cap:
        chunk = await stream.read(65536)
        if not chunk:
            return bytes(buf), False
        buf += chunk
    del buf[cap:]
    return bytes(buf), True


def _too_large(size: int | None, max_bytes: int) -> ValueError:
    measured = f" ({size} bytes)" if size is not None else ""
    return ValueError(
        f"Image is too large{measured}. The maximum is {max_bytes} bytes."
    )


class ViewImageFileConfig(BaseModel):
    max_img_dim: int = 2000
    # Enforces the docstring's 5 MB limit; base64 otherwise buffers the whole
    # (model-controlled) file into the root server process.
    max_file_bytes: int = 5 * 1024 * 1024


@final
class view_image_file(ToolBase[ViewImageFileConfig]):
    config_schema = ViewImageFileConfig

    async def __call__(self, file_path: Path) -> ToolResult:
        """
        Views the image files at absolute path file_path.

        The image must be a .jpeg, .png, .gif, or .webp and must be at most 5MB.
        """
        try:
            from PIL import Image as PILImage
        except ImportError:
            raise ImportError(
                "Pillow is required to use the view_image_file tool. "
                + "Install it with `uv add Pillow`."
            )

        if not file_path.is_absolute():
            raise ValueError(f"File path must be absolute: {str(file_path)!r}")

        if not file_path.is_file():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Fast honest-path gate; the capped read below stays bounded even when
        # a race defeats this check.
        size = os.stat(file_path).st_size
        if size > self.config.max_file_bytes:
            raise _too_large(size, self.config.max_file_bytes)

        proc = await asyncio.create_subprocess_exec(
            BASE64_PATH,
            "-w",
            "0",
            preexec_fn=open_regular_as_stdin(str(file_path), make_demote_fn()),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        assert proc.stdout is not None
        assert proc.stderr is not None

        # +1 for the trailing newline base64 emits even with -w 0.
        output_cap = _b64_encoded_len(self.config.max_file_bytes) + 1
        stderr_task = asyncio.create_task(
            _drain_bounded(proc.stderr, _MAX_STDERR_BYTES)
        )
        try:
            async with asyncio.timeout(demoted.SUBPROCESS_TIMEOUT_S):
                stdout, hit_cap = await _read_capped(proc.stdout, output_cap)
                if hit_cap:
                    await demoted.reap(proc)
                try:
                    stderr = await stderr_task
                except Exception:  # noqa: BLE001 - stderr is diagnostic only
                    stderr = b""
                await proc.wait()
        except TimeoutError:
            raise RuntimeError(f"Timed out reading image {file_path}") from None
        finally:
            stderr_task.cancel()
            await demoted.reap(proc)

        if err := open_error(file_path, proc.returncode, stderr):
            raise err

        if hit_cap:
            raise _too_large(None, self.config.max_file_bytes)

        if proc.returncode != 0:
            raise RuntimeError(
                f"Failed to read image: {stderr.decode('utf-8', errors='replace')}"
            )

        image_b64_bytes = base64.b64decode(stdout)
        img = PILImage.open(io.BytesIO(image_b64_bytes))

        if img.width > self.config.max_img_dim or img.height > self.config.max_img_dim:
            raise ValueError(
                "Image is too large. Please try a smaller image. Width and height "
                + f"of image must be less than {self.config.max_img_dim} pixels."
            )

        img = img.convert("RGB")
        buffered = io.BytesIO()
        img.save(buffered, format="jpeg")
        encoded = buffered.getvalue()
        if len(encoded) > self.config.max_file_bytes:
            raise _too_large(len(encoded), self.config.max_file_bytes)

        img = FastMCPImage(data=encoded, format="jpeg")
        return ToolResult(content=[img.to_image_content()])
