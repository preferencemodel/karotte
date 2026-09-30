from typing import Literal

Runtime = Literal["podman", "docker", "docker:gvisor", "nerdctl"]
Engine = Literal["podman", "docker", "nerdctl"]


def get_engine(runtime: Runtime) -> Engine:
    """Get the container engine command from a runtime string.

    For compound runtimes like ``"docker:gvisor"``, returns the engine
    portion (``"docker"``).
    """
    engine = runtime.split(":")[0]
    assert engine in ("podman", "docker", "nerdctl")
    return engine  # type: ignore[return-value]
