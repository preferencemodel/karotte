# Virtual Environments

Each subdirectory defines a virtual environment that gets built into the container.

## Access levels

| `access`     | Effect                                   | Recommended path                              |
| ------------ | ---------------------------------------- | --------------------------------------------- |
| `root`       | Student cannot access.                   | `/root/venvs/<name>/` (hidden from student) |
| `student:r`  | Student can read/execute but not modify. | `/workdir/venvs/<name>/`                      |
| `student:rw` | Student has full access.                 | `/workdir/.venv`                              |

Root-only venvs should use paths under `/root/` so the student cannot even see they exist.

## Adding a venv

1. Create a subdirectory with a `pyproject.toml`.
2. Run `just lock-venvs` to generate the lockfile.

Example `pyproject.toml` for a root-only scoring venv:

```toml
[project]
name = "scoring-env"
version = "0.1.0"
requires-python = "==3.12.*"
dependencies = ["rich"]

[tool.karotte]
access = "root"
path = "/root/venvs/scoring"
```
