# Tasks and steps

A task is a Python class that tells karotte what the student sees, which tools it gets, and how its work is scored.
A task has one or more steps.
Each step sends the student instructions and ends with a judge that scores the result.

## A minimal task

```python
import re

from karotte import Step, Task
from karotte.judges import RegexJudge


class FindPython(Task):
    id = "find-python"

    @property
    def system_prompt(self) -> str:
        return "You are working in a Linux shell."

    @property
    def steps(self):
        return [FindPythonStep(config=self.config)]

    @property
    def tools(self):
        return ["bash"]


class FindPythonStep(Step):
    @property
    def instructions(self) -> str:
        return "Find the path to your Python executable. Answer with `path: <path>`."

    @property
    def judge(self):
        return RegexJudge([re.compile(r"path: .*/python3?")])
```

The student gets the system prompt and the step's instructions, works on them with the `bash` tool, and the judge checks its final answer.
See [Scoring](scoring.md) and [Tools](tools.md).

## Task packages

Each task is a package in `src/environment/tasks/`: a directory whose `__init__.py` defines the `Task` class.
Create one with:

```sh
uv run just create-task my-task
```

This copies `tasks/_template/` to `tasks/my_task/` and sets the class's `id` to `my-task`.
The `example-task` task in the `default` template is a working example with two steps, a submission file, hints and a scoring script.

A task's `id` must be unique within the environment and at most 255 characters long.
`uv run karotte tasks list` shows every id, and the run config's `task_id` picks the task to run.

## Discovery

karotte loads tasks by calling `get_tasks()` in `src/environment/__init__.py`.
It imports every directory in `tasks/` whose name doesn't start with `_`, and collects every `Task` subclass in that package's namespace that has a non-empty `id`.

To load only some tasks, fill in the two sets in `src/environment/__init__.py`:

```python
INCLUDE_TASKS: set[str] = {"example-task", "my-task"}
EXCLUDE_TASKS: set[str] = set()
```

## The Task class

A task subclasses `karotte.Task`.
karotte creates it with the run config, which the task reads as `self.config`, so any property can depend on the run.

| Member              | Kind            | Default                       | Purpose                                                                  |
| ------------------- | --------------- | ----------------------------- | ------------------------------------------------------------------------ |
| `id`                | class attribute | required                      | The task's id.                                                           |
| `system_prompt`     | property        | required                      | The system message, or `None` for none.                                  |
| `steps`             | property        | required                      | The steps, in order.                                                     |
| `tools`             | property        | required                      | Names of the tools the student gets. See [Tools](tools.md).              |
| `required_hardware` | property        | the plugins' default          | See [Required hardware](#required-hardware).                             |
| `configure_tools()` | method          | does nothing                  | See [Hooks](#hooks).                                                     |
| `pre_hook()`        | method          | returns `{}`                  | See [Hooks](#hooks).                                                     |
| `submission_paths`  | property        | all steps' `submission_paths` | Every path the student hands in.                                         |
| `data_mounts`       | property        | `[]`                          | Read-only data a [backend](../extending/backend.md) attaches at runtime. |

Declare the properties with `@property`; overriding one with a plain method raises a `TypeError` when the class is defined.

!!! note

    `karotte build` and `karotte tasks list --json` create every task with a placeholder run config (empty `run_id`, `task_id` and `model`) to read `tools`, `required_hardware`, `submission_paths` and `data_mounts`.
    These properties, and `steps`, which `submission_paths` reads, must work with that config.

CLI agents send their own system prompt instead of the task's; see [Run config](../running/run-config.md#cli-agents).

## The Step class

A step subclasses `karotte.Step`.
Create steps with the task's config, as in `FindPythonStep(config=self.config)`, so they can read `self.config` too, for example `self.config.use_hints` to add hints to the instructions.

| Member               | Kind                        | Default      | Purpose                                                                    |
| -------------------- | --------------------------- | ------------ | -------------------------------------------------------------------------- |
| `instructions`       | property                    | required     | The user message that starts the step.                                     |
| `judge`              | property                    | required     | The [judge](scoring.md) that scores the step.                              |
| `submission_paths`   | property or class attribute | `None`       | Where the student writes its answer files, as a tuple of absolute `Path`s. |
| `pre_scoring_hook()` | method                      | does nothing | See [Hooks](#hooks).                                                       |
| `post_hook()`        | method                      | does nothing | See [Hooks](#hooks).                                                       |

karotte reads `judge` after `pre_scoring_hook` has run, so the judge can use what the hook prepared, such as the saved copy of a submission.
See [Saving submissions](scoring.md#saving-submissions) for the full pattern.

## Hooks

| Hook                      | Runs                                                                                                          | Use it to                                                                                                                                                                 |
| ------------------------- | ------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `Task.configure_tools()`  | Before the tool server registers the task's tools.                                                            | [Configure tools](tools.md#configuring-tools).                                                                                                                            |
| `Task.pre_hook()`         | Once, before the first step, after the default [student limits](../running/student-resources.md) are applied. | Generate or copy task data, adjust limits with `limit_resources`, write values for the grader to the `ProtectedStore`. The dict it returns is recorded in the transcript. |
| `Step.pre_scoring_hook()` | After the student finishes the step, before the judge.                                                        | Stop the student's processes and collect its submission. Tasks from the `default` template call `collect_submission` here.                                                |
| `Step.post_hook()`        | After the step is scored, also when it ends the run.                                                          | Clean up.                                                                                                                                                                 |

## Extra config

The run config's `extra_config` is a free-form dict passed to your task unchanged, so you can run the same image with different settings without a rebuild.
Read it as `self.config.extra_config`:

```python
@property
def instructions(self) -> str:
    difficulty = (self.config.extra_config or {}).get("difficulty", "easy")
    ...
```

A few keys, such as `extra_task_instructions`, have a built-in meaning; see [Run config](../running/run-config.md#extra-config).

## Required hardware

`required_hardware` names the hardware a task needs.
karotte itself knows no hardware names.
It is up to a [plugin](../extending/plugins.md#karottehardware_limits) to define them and decide what a sandbox on that hardware should look like.

```python
@property
def required_hardware(self) -> str | None:
    return "my-gpu"
```
