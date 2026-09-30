import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_IN_CONTAINER = "KAROTTE_CONTAINERIZED" in os.environ


def _container_or_local(container_path: str, local_subdir: str) -> Path:
    return Path(container_path) if _IN_CONTAINER else PROJECT_ROOT / local_subdir


STUDENT_WORKDIR = _container_or_local(container_path="/workdir", local_subdir=".")

STUDENT_DATA_DIR = _container_or_local(
    container_path="/workdir/data", local_subdir="student_data"
)
SHARED_DATA_DIR = _container_or_local(
    container_path="/workdir/shared", local_subdir="shared_data"
)
ROOT_DATA_DIR = _container_or_local(
    container_path="/root_data", local_subdir="root_data"
)
INTERMEDIATE_DATA_DIR = _container_or_local(
    container_path="/intermediate_data", local_subdir="intermediate_data"
)
