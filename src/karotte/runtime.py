from typing import Literal

Runtime = Literal["podman", "docker", "docker:gvisor", "nerdctl", "apple-container"]
Engine = Literal["podman", "docker", "nerdctl", "container"]


def get_engine(runtime: Runtime) -> Engine:
    """Get the container engine command from a runtime string.

    For compound runtimes like ``"docker:gvisor"``, returns the engine
    portion (``"docker"``).
    """
    if runtime == "apple-container":
        # Apple's CLI is called `container`.
        return "container"
    engine = runtime.split(":")[0]
    assert engine in ("podman", "docker", "nerdctl", "container")
    return engine  # type: ignore[return-value]
