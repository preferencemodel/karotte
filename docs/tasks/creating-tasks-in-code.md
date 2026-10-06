# Creating tasks in code

Defining each task by hand works well if you environment contains a small number of distinct tasks.
For dozens of variations of one task, generate the classes instead.
Karotte has a `create_task` factory for simple cases, and the `default` template has a scaffold for task suites driven by a JSON file.

## create_task

`create_task` builds a `Task` subclass from arguments.
Put the calls in a task package's `__init__.py` and assign each class to a module-level name, since [discovery](tasks-and-steps.md#discovery) collects the task classes in the package's namespace:

```python
import re

from karotte import StepConfig, create_task
from karotte.judges import RegexJudge

QUESTIONS = {
    "add-small": ("What is 2 + 3?", "5"),
    "add-large": ("What is 1234 + 4321?", "5555"),
}

for task_id, (question, answer) in QUESTIONS.items():
    globals()[task_id.replace("-", "_")] = create_task(
        id=task_id,
        tools=["bash"],
        system_prompt="You are working in a Linux shell.",
        steps=[
            StepConfig(
                instructions=f"{question} Answer with `answer: <number>`.",
                judge=RegexJudge([re.compile(rf"answer: {answer}\b")]),
            )
        ],
    )
```

`create_task` also takes the task's optional members, such as `pre_hook` and `required_hardware`, and `StepConfig` takes the step's.
[`task_factory.py`](https://github.com/preferencemodel/karotte/blob/main/src/karotte/task_factory.py) lists them all.

## Task suites

Once the variations span more than one axis, or tasks need their own scoring parameters, a loop in one file gets unwieldy.
A task suite keeps the task definitions in committed JSON files and the shared behavior in one class.

```sh
uv run just create-task-suite my-suite
```

This copies `tasks/_template_suite/` to `tasks/my_suite/`, sets the suite's id to `my-suite`, and generates the JSON.

| File                                         | Role                                                                                                 |
| -------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| `task_configs/task_config.py`                | The `TaskConfig` and `TaskCalibration` schemas. Add one `TaskConfig` field per axis of variation.    |
| `prompt.py`                                  | `build_instructions(...)` writes the instruction text from a task's axis values.                     |
| `task_configs/create_task_configs.py`        | Loops over the axes and writes `task_configs.json`. Task ids are `<suite id>-<variant>`.             |
| `task_configs/create_calibration_configs.py` | Writes each task's scoring parameters to `calibration_configs.json`.                                 |
| `task_configs/*.json`                        | Generated and committed. The suite reads these at runtime.                                           |
| `__init__.py`                                | The shared `SuiteTask` and `SuiteStep`. `register_suite` creates one task class per row of the JSON. |
| `scoring_script.py`                          | The grader, run by `ExecutableJudge`. Reads the active task's config back and writes a score.        |

After changing the axes or the instruction text, regenerate the JSON:

```sh
uv run just gen-configs my-suite
```

Adding, removing or changing a task is a data change: edit the generators and regenerate.
Don't edit the generated JSON by hand.

Instructions and submission paths can contain the literal placeholder `{STUDENT_WORKDIR}`.
`SuiteStep` fills it in when the task runs, so the same JSON works locally and in the container.

The suite's `pre_hook` hands the active task's config and calibration to the grader with `stash_task_configs("my_suite", ...)`, which writes them to the `ProtectedStore` (see [Data and dependencies](data-and-dependencies.md)).
The scoring script reads them back with `load_task_config` and `load_task_calibration` from `environment.suite`.
