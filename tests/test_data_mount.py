import pytest
from pydantic import ValidationError

from karotte.schemas import DataMount


def test_data_mount_explicit_paths():
    mount = DataMount(name="cifar-10", version="v1", container_path="/data/cifar")
    assert mount.name == "cifar-10"
    assert mount.version == "v1"
    assert mount.container_path == "/data/cifar"
    assert mount.mount_type == "read"


def test_data_mount_round_trips_as_json():
    """`tasks list --json` output should round-trip cleanly."""
    mount = DataMount(name="x", version="v1", container_path="/x")
    dumped = mount.model_dump()
    assert dumped == {
        "name": "x",
        "version": "v1",
        "container_path": "/x",
        "mount_type": "read",
    }
    reloaded = DataMount.model_validate(dumped)
    assert reloaded.model_dump() == dumped


def test_data_mount_requires_container_path():
    with pytest.raises(ValidationError):
        DataMount(name="x", version="v1")  # pyright: ignore[reportCallIssue]
