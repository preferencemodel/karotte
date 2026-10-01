"""Run karotte in a Firecracker microVM booted from the environment's image."""


class FirecrackerError(RuntimeError):
    """Why a Firecracker VM couldn't be set up or started."""
