# Tools

The student interacts with the environment through tools.
Karotte serves them over the [Model Context Protocol](https://modelcontextprotocol.io/) (MCP) from a server that runs inside the sandbox.

## Built-in tools

Karotte ships these tools in `karotte.tools`:

| Tool                 | What it does                                                                                                                            |
| -------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| `bash`               | Runs a command in a persistent bash session as the student.                                                                             |
| `view_lines_in_file` | Returns a range of lines from a file.                                                                                                   |
| `replace_in_file`    | Replaces an exact string in a file and returns a diff.                                                                                  |
| `view_image_file`    | Shows the model a `.jpeg`, `.png`, `.gif` or `.webp` image.                                                                             |
| `validate`           | Scores the student's current submission on a dev split and reports how it did; see [Validating a submission](#validating-a-submission). |

The file tools check permissions and read or write the file as the student, so the student can't use them to reach files it couldn't reach from `bash`.

## Picking tools for a task

A task returns tool names from its `tools` property:

```python
@property
def tools(self):
    return ["bash", "view_lines_in_file", "replace_in_file"]
```

CLI agents that bring their own shell and file tools use those instead of Karotte's.

## Configuring tools

Tools that subclass `ToolBase` read a config when they start.
Write one with `ToolConfigWriter` in the task's `configure_tools()`, which runs before the MCP server registers the task's tools:

```python
from karotte import Task, ToolConfigWriter
from karotte.tools.bash import BashConfig


class MyTask(Task):
    def configure_tools(self) -> None:
        ToolConfigWriter().write("bash", BashConfig(default_timeout_s=600))
```

`write()` returns the writer, so calls chain.
Configs are stored as JSON in `~/.config/karotte/tool_configs/` of the user running the run, which is root inside the sandbox.

## Custom tools

Put a custom tool in `src/environment/tools/<name>.py`.
Karotte automatically finds it by name; no need to register the tool.
The module must define an object called `<name>` that is either:

- an async or regular function, or
- a class with a `__call__` method.
  Karotte creates one instance per run, with no arguments.

The function or `__call__` needs

- a docstring, which becomes the tool description,
- type annotations on every parameter,
- and a return annotation of `ToolResult`.

Raise an exception to report an error; the model sees the error message.
The `builtin` and `external` agents pass only the first content block of a result to the model.

A tool that records the student's answer and saves it in a place where the student can't change it:

```python
# src/environment/tools/submit_answer.py
from fastmcp.tools.tool import ToolResult
from karotte import ProtectedStore


async def submit_answer(answer: str) -> ToolResult:
    """Submit your final answer."""
    ProtectedStore().write("answer", answer)
    return ToolResult(content="Answer recorded.")
```

To make a tool configurable, subclass `ToolBase` with a pydantic model as its config.
The class name is the tool name, and `self.config` holds the config:

```python
# src/environment/tools/run_tests.py
import asyncio

from fastmcp.tools.tool import ToolResult
from karotte import ToolBase
from karotte.subprocess import make_demote_fn, student_env
from pydantic import BaseModel


class RunTestsConfig(BaseModel):
    pytest_args: list[str] = []


class run_tests(ToolBase[RunTestsConfig]):
    config_schema = RunTestsConfig

    async def __call__(self, path: str) -> ToolResult:
        """Run the tests in `path` and return their output."""
        proc = await asyncio.create_subprocess_exec(
            "/workdir/.venv/bin/python",
            "-m",
            "pytest",
            *self.config.pytest_args,
            path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=student_env(),
            preexec_fn=make_demote_fn(1),
        )
        stdout, _ = await proc.communicate()
        return ToolResult(content=stdout.decode())
```

A task then configures it with `ToolConfigWriter().write("run_tests", RunTestsConfig(pytest_args=["-x"]))`.

A custom tool with the same name as a built-in one replaces it.

`karotte check` lists every tool it finds and fails if one doesn't meet these rules.
Make sure that the tool's dependencies are in the root venv (see [Python dependencies](data-and-dependencies.md#python-dependencies)).

## Validating a submission

The `validate` tool lets the student check its submission part-way through a step.
It takes no arguments: it always grades the files in the step's `submission_paths`, as they are at that moment.
It never scores the step, so the step's judge still decides the score.

Each call copies the submission to a place only root can read, runs a _dev scorer_ on the copy as root, and returns one JSON object:

| Field                 | Meaning                                                                                                        |
| --------------------- | -------------------------------------------------------------------------------------------------------------- |
| `tier`                | `valid` (graded), `no_valid_output` (missing, or rejected by the scorer) or `crash` (the scorer failed on it). |
| `dev_metric`          | The dev score to three significant figures, or `null` unless `tier` is `valid`.                                |
| `metric`              | The metric's name and which direction is better.                                                               |
| `reason`              | Why a call has no score, or `null`. Text from the scorer is cleaned and cut; see the scorer contract below.    |
| `delta_vs_prev_valid` | `dev_metric` minus the last valid call's, both as shown to three significant figures, or `null`.               |
| `best_dev_metric`     | The best `dev_metric` so far, or `null`.                                                                       |
| `elapsed_s`           | How long the call took.                                                                                        |
| `calls_left`          | Calls the student has left.                                                                                    |

A call is refused with `reason` set, and costs nothing, once `max_calls` calls have been made (`"budget spent"`) or when less than `min_interval_s` has passed since the last one.
A refused call reports the tier of the last call and `dev_metric: null`.

The tool keeps one state for the whole task: its calls, their spacing and the best copy.
In a task with several steps it grades the same `submission_paths` whichever step the student is in, so give such a task one `ValidateConfig` that lists the paths of every step, and don't use the run config's [submission fallbacks](../running/run-config.md#submission-fallbacks), which refuse a task with more than one step.

Configure the tool in `configure_tools()` and add it to the task's `tools`:

```python
import sys

from karotte import Task, ToolConfigWriter
from karotte.tools.validate import ValidateConfig


class MyTask(Task):
    @property
    def tools(self):
        return ["bash", "validate"]

    def configure_tools(self) -> None:
        ToolConfigWriter().write(
            "validate",
            ValidateConfig(
                dev_command=[sys.executable, "-m", "environment.dev_score"],
                metric="accuracy",
                submission_paths=self.submission_paths,
                max_calls=5,
                min_interval_s=60,
                timeout_s=60,
            ),
        )
```

`higher_is_better` (default `true`) says which direction the metric improves in.
`snapshot_limits` takes the limits `save_submission` takes (`max_file_bytes`, `max_total_bytes`, `max_entries`, `max_depth`); pass the ones the step's final judge copies with.
`submission_paths` need distinct file names.

Karotte runs the dev scorer as `dev_command` plus two arguments: a directory holding a copy of each submission path under its own file name, and the path of a JSON file to write.
The file has the shape an `ExecutableJudge` script writes, with two optional keys:

```json
{ "score": 0.83, "metadata": {}, "tier": "valid", "reason": null }
```

- `score` is required when the tier is `valid`, which is the default.
- `tier` may also be `no_valid_output` or `crash`; then `reason` is shown to the student.
  Karotte removes control characters from it, collapses runs of whitespace (line breaks included) into one space, and cuts it to 300 characters.
  It is the only free text from the scorer that reaches the student, so never put labels, expected outputs or details of the dev split in it: say what is wrong with the submission, such as `expected 10 columns, found 8`.
- Nothing else reaches the student: not `metadata`, and not the scorer's output.

A scorer that exits with an error, writes no usable file, or runs past `timeout_s` counts as a `crash`.
Karotte starts it in its own process group at lower priority and kills the group when the call ends.
It doesn't stop the student's processes.

Keep the dev split out of the final judge's data, and out of the student's reach.
Otherwise repeated calls reveal the labels the final score depends on.

Karotte keeps the copy with the best valid score.
Two [run config](../running/run-config.md#submission-fallbacks) options can score it in place of what the student left.

## Running commands as the student

!!! warning

    The MCP server runs as root.
    A custom tool that starts a subprocess starts it as root unless it drops privileges.
    The `bash` and file tools switch to the student user (uid and gid 1000 in the `default` template) first.

Pass `make_demote_fn()` from `karotte.subprocess` as `preexec_fn` to run the child as the student:

```python
from karotte.subprocess import make_demote_fn, student_env

proc = await asyncio.create_subprocess_exec(
    "my-command",
    env=student_env(),
    preexec_fn=make_demote_fn(),
)
```

The child takes its uid and gid from `KAROTTE_DEMOTE_ID`.

Demotion changes only the uid and gid.
The child still inherits the server's environment.
`student_env()` returns that environment without secrets and with `HOME`, `USER` and `LOGNAME` set for the student.

### Handing over pipes

If you give the child pipes (`stdin`, `stdout` or `stderr` set to `PIPE`), name the child fds they become so they are handed to the student too:

```python
proc = await asyncio.create_subprocess_exec(
    "my-command",
    stdout=asyncio.subprocess.PIPE,
    stderr=asyncio.subprocess.PIPE,
    # fd 1 and fd 2 in the child are the pipes opened above.
    preexec_fn=make_demote_fn(1, 2),
)
```

A pipe created this way belongs to the MCP server, which is root.
Without the handover the child can read and write the fds it inherited, but can't reopen them by path (`/dev/stdin`, `/dev/stdout`, `/proc/self/fd/N`), which is how many programs reach their own standard streams.

### New sessions

`make_preexec()` takes the same arguments and also starts a new session (`os.setsid()`) before dropping privileges.
Use it when you want to kill the child's whole process group later.
