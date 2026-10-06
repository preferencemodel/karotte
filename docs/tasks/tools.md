# Tools

The student interacts with the environment through tools.
Karotte serves them over the [Model Context Protocol](https://modelcontextprotocol.io/) (MCP) from a server that runs inside the sandbox.

## Built-in tools

Karotte ships these tools in `karotte.tools`:

| Tool                 | What it does                                                |
| -------------------- | ----------------------------------------------------------- |
| `bash`               | Runs a command in a persistent bash session as the student. |
| `view_lines_in_file` | Returns a range of lines from a file.                       |
| `replace_in_file`    | Replaces an exact string in a file and returns a diff.      |
| `view_image_file`    | Shows the model a `.jpeg`, `.png`, `.gif` or `.webp` image. |

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

A tool that records an answer from the student answer and saves it in a place where the student can't change it:

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
