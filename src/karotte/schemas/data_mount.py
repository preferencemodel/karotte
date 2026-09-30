"""Schema for a data mount declared on a Task."""

from typing import Literal

from pydantic import BaseModel

MountType = Literal["read"]
"""Access mode for a data mount.

``"read"`` is currently the only value. Future access modes (e.g. ``"write"``)
may be added.
"""


class DataMount(BaseModel):
    """A data mount declared on a Task.

    Pins to an immutable ``(name, version)`` entry in the backend's mount
    registry and declares where the data should be surfaced inside the container.

    Data mounts are only materialized in containerized runs on a backend.
    Tasks that need to read the data in non-containerized local dev should
    branch on ``karotte.container.is_containerized()`` and provide their own
    local fallback.
    """

    name: str
    """Name of the mount in the registry (e.g., 'cifar-10')."""

    version: str
    """Version of the mount in the registry (e.g., 'v1')."""

    container_path: str
    """Absolute path inside the container where the mount is surfaced."""

    mount_type: MountType = "read"
    """Access mode for this mount. Only ``"read"`` is supported today."""
