from typing import Literal

Runtime = Literal["podman", "docker", "docker:gvisor", "apple-container", "firecracker"]
Engine = Literal["podman", "docker", "container"]


def get_engine(runtime: Runtime) -> Engine:
    """Get the container engine command from a runtime string.

    For compound runtimes like ``"docker:gvisor"``, returns the engine
    portion (``"docker"``).
    """
    if runtime == "apple-container":
        # Apple's CLI is called `container`.
        return "container"
    if runtime == "firecracker":
        # A Firecracker VM boots from an image docker builds.
        return "docker"
    engine = runtime.split(":")[0]
    assert engine in ("podman", "docker", "container")
    return engine  # type: ignore[return-value]
