# Plugins

A Python package can extend karotte through [entry points](https://packaging.python.org/en/latest/specifications/entry-points/).
karotte reads them from every package installed next to it.
Templates have their own entry point, which [Template packages](template-packages.md) describes.

| Entry point                        | Points at                                          | Effect                                                                                                                |
| ---------------------------------- | -------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| `karotte.cli`                      | a Typer app                                        | Adds a subcommand named after the entry point.                                                                        |
| `karotte.run_config_preprocessors` | `f(config) -> config`                              | Rewrites the run config before a run.                                                                                 |
| `karotte.default_proxy_url`        | a string                                           | The default for `karotte run --proxy`.                                                                                |
| `karotte.harness_secret_env`       | a list of names                                    | Environment variables hidden from the student.                                                                        |
| `karotte.platform_tooling_dirs`    | a list of paths                                    | Directories where a platform mounts its own tooling into every container. They are hidden from graded toolchain runs. |
| `karotte.age_delay_exemptions`     | a list of package names                            | Extra packages exempt from uv's `exclude-newer` delay.                                                                |
| `karotte.update_migrations`        | an object with `prepare`, `tool` and `migrate`     | During `karotte update`, moves environments made by an older release to the current names.                            |
| `karotte.default_hardware`         | a string                                           | The `required_hardware` of tasks that don't set one.                                                                  |
| `karotte.hardware_limits`          | `f(hardware)` returning `HardwareLimits` or `None` | The memory, disk and CPUs a sandbox on that hardware gets.                                                            |
| `karotte.container_run_args`       | `f(task, runtime) -> list[str]`                    | Extra arguments for the container engine's `run` command.                                                             |

## Registering a plugin

Declare entry points in your package's `pyproject.toml`.
Each value has the form `module:attribute`.
karotte loads the attribute and uses it as described in the table above.

```toml
[project.entry-points."karotte.harness_secret_env"]
my_plugin = "my_plugin:HARNESS_SECRETS"

[project.entry-points."karotte.run_config_preprocessors"]
my_plugin = "my_plugin:preprocess"
```

```python
from karotte.schemas.evaluation_run_config import EvaluationRunConfig

HARNESS_SECRETS = ["MY_SERVICE_TOKEN"]


def preprocess(config: EvaluationRunConfig) -> EvaluationRunConfig:
    return config.model_copy(update={"save_artifacts": False})
```

Install the package into the same environment as karotte.
For an environment's venv, add it as a dependency.
For a tool install, use `uv tool install karotte --with my-plugin` or `uvx --with my-plugin karotte ...`.

## Order and merging

If only one value can win, karotte sorts the entry points by name and takes the first one that loads.
If the values add up, karotte uses all of them.

| Entry point                                                                                   | When several are installed                                                            |
| --------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| `karotte.cli`                                                                                 | Each one adds its subcommand.                                                         |
| `karotte.run_config_preprocessors`                                                            | All of them run, in entry point name order. Each one gets the previous one's output.  |
| `karotte.default_proxy_url`                                                                   | The first by name wins.                                                               |
| `karotte.harness_secret_env`, `karotte.platform_tooling_dirs`, `karotte.age_delay_exemptions` | karotte merges the lists.                                                             |
| `karotte.update_migrations`                                                                   | All of them run. For `tool`, the first one that doesn't return `None` wins.           |
| `karotte.default_hardware`, `karotte.hardware_limits`                                         | The first by name wins.                                                               |
| `karotte.container_run_args`                                                                  | All of them run, in entry point name order, and karotte concatenates their arguments. |

## When a plugin fails to load

If a plugin fails to load, karotte skips it with a warning and carries on without it.
Run config preprocessors are the exception.
If one of them fails to load or raises an exception, the run fails.

## The entry points

### `karotte.cli`

The attribute is a `typer.Typer` app.
It becomes the subcommand `karotte <entry point name>`.
If an entry point has the same name as a built-in command, karotte ignores it with a warning.
Plugins never replace karotte's own commands.

```toml
[project.entry-points."karotte.cli"]
mytool = "my_plugin.cli:app"
```

### `karotte.run_config_preprocessors`

The attribute is a function that takes the parsed [`EvaluationRunConfig`](https://github.com/preferencemodel/karotte/blob/main/src/karotte/schemas/evaluation_run_config.py) and returns one.
`karotte run` calls it right after it parses `--config`, before it picks the runtime or loads the task.
See [Run config](../running/run-config.md).

### `karotte.default_proxy_url`

The attribute is a string.
It's the URL that `karotte run` sends model calls to when you don't pass `--proxy`.
If you pass `--no-proxy`, karotte ignores it.
See [Run config](../running/run-config.md) for how proxies work.

### `karotte.harness_secret_env`

The attribute is a list of environment variable names.
karotte leaves these variables out of the environment it gives the student's processes, such as `bash` tool calls and CLI agents.
Use it for credentials that the harness needs but the student must not see.
karotte hides only the names you list, because tasks sometimes hand the student a secret on purpose.

### `karotte.platform_tooling_dirs`

The attribute is a list of directories where the platform running karotte mounts its own tooling into every container.
During builds and graded runs, the [`language-toolchains`](../environments/language-toolchains.md) template covers each of them with an empty, read-only tmpfs.
That way a submission can't load an interpreter from them.
Paths that don't exist are ignored.

### `karotte.age_delay_exemptions`

The attribute is a list of package names.
When karotte runs uv outside a project (for `karotte update` and `post_create.py`), it applies a 7-day `exclude-newer` delay.
karotte itself and the packages in these lists are exempt from that delay.
Inside an environment, the delay comes from the environment's own `pyproject.toml`.

### `karotte.update_migrations`

The attribute is an object with three methods.
`karotte update` calls them as follows:

| Method                                            | Called                                                                                          |
| ------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| `prepare(project_dir)`                            | Before karotte reads the environment's manifest.                                                |
| `tool(version)`, returning `list[str]` or `None`  | To get the `uv tool run` arguments that run an older release. `None` means `karotte@<version>`. |
| `migrate(baseline_dir, project_dir, old_version)` | Before the merge, on the old release's render and on the environment.                           |

Use it when a release renames things that the 3-way merge in [Updating environments](../environments/updating.md) can't follow on its own.

### `karotte.default_hardware`

The attribute is a string.
It's the hardware name for tasks that don't set `required_hardware`.
karotte itself doesn't know any hardware names.
Plugins define them.
See [Tasks and steps](../tasks/tasks-and-steps.md) for the task property.

### `karotte.hardware_limits`

The attribute is a function that takes a hardware name and returns a `karotte.hardware.HardwareLimits`.
For hardware it doesn't know, it returns `None`.

| Field          | Meaning                                                                                                                                      |
| -------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| `memory_bytes` | The RAM the sandbox gets. The student gets this amount minus 1 GiB, which is left for the harness.                                           |
| `disk_bytes`   | The cap on the student's disk quota. It doesn't apply if the launcher sets `KAROTTE_DISK_BUDGET_BYTES`.                                      |
| `cpus`         | The number of CPUs a VM runtime gives the sandbox.                                                                                           |
| `passthrough`  | `True` for hardware that a VM can't provide, such as a GPU. Tasks on this hardware run under docker by default, and VM runtimes refuse them. |

```python
from karotte.hardware import HardwareLimits

GIB = 1024**3


def hardware_limits(hardware: str) -> HardwareLimits | None:
    if hardware == "cpu-large":
        return HardwareLimits(memory_bytes=32 * GIB, disk_bytes=100 * GIB, cpus=8)
    if hardware == "gpu":
        return HardwareLimits(passthrough=True)
    return None
```

A hardware name that no plugin knows is not an error.
karotte just uses its defaults.
Without an answer from a plugin, a VM gets 2 CPUs and 4 GiB for the sandbox.
On other runtimes, the sandbox's memory comes from `KAROTTE_SANDBOX_MEMORY_BYTES`.
If that variable isn't set, it comes from the sandbox's cgroup limit or RAM.
The student gets this amount minus 1 GiB.
If the variable isn't set and there's no working cgroup, the student has no memory limit.
See [Runtimes](../running/runtimes.md) and [Student resources](../running/student-resources.md).

### `karotte.container_run_args`

The attribute is a function that takes the task and the runtime name (`docker`, `podman`, `nerdctl`, ...).
It returns extra arguments for the engine's `run` command.
You can use it to pass devices through, for example.
karotte adds these arguments for `docker`, `podman`, `docker:gvisor` and `nerdctl`.

`karotte run` calls every hook before it launches, whatever the runtime.
If a hook raises an exception, karotte refuses to launch and shows the exception's message.

```python
def container_run_args(task, runtime: str) -> list[str]:
    if task.required_hardware != "gpu":
        return []
    if runtime != "docker":
        raise RuntimeError("GPU tasks need --runtime docker")
    return ["--gpus", "all"]
```
