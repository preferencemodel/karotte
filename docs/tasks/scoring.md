# Scoring

Every step ends with a judge that scores the student's work.
This page covers the judges Karotte ships, how to combine them, how to write your own, and how to collect files the student hands in.

## How a step gets scored

When the student finishes a step, Karotte calls the step's `pre_scoring_hook()`, then reads its `judge` property and calls `judge.evaluate(transcript)`.
The transcript holds every event of the run so far.
The judge returns a `Scoring`:

```python
from karotte.schemas.scoring import Scoring

Scoring(score=1.0, metadata={"answer": "42"}, continue_task=True)
```

The run's score is the score of the last step that was scored.

## Built-in judges

Import them from `karotte.judges`.

### ExecutableJudge

`ExecutableJudge` runs a command, usually a scoring script, and reads the score it writes.
It doesn't look at the transcript.

```python
import sys

from karotte.judges import ExecutableJudge

ExecutableJudge(
    [
        sys.executable,
        "-m",
        "environment.tasks.example_task.scoring_script",
        str(self.saved_submissions[0]),
        "score_output.txt",
    ]
)
```

- The first argument is the full command.
  Use `sys.executable` so the script runs with the environment's Python.
- The last argument is the name of the file the script writes its result to.
  Karotte puts the file in a fresh temporary directory, passes the script that path, and deletes the directory after scoring finishes.
- The script must write `{"score": <float>, "metadata": {...}}` to that file, with both keys.
  Karotte converts the metadata values to strings and adds the script's `stdout` and `stderr`.
- The script must exit 0.
  A non-zero exit, a missing executable, or an output file that is missing or not valid JSON scores 0 with `continue_task=False` and the error in the scoring metadata.
- `continue_threshold` (default `-1`): the task continues if the score is at least this.
  With the default, any non-negative score continues the task.
- `cwd` sets the script's working directory.

The scoring script of the `example-task` task:

```python
import json
import sys
from pathlib import Path

if __name__ == "__main__":
    submission_path, output_path = sys.argv[1], sys.argv[2]

    answer = Path(submission_path).read_text().strip()

    score = 1.0 if "3.12.11" in answer else 0.0

    Path(output_path).write_text(json.dumps({"score": score, "metadata": {}}))
```

### RubricJudge

`RubricJudge` asks an LLM whether the student's work meets each criterion of a rubric.
The score is the sum of the weights of the criteria it meets.

```python
from karotte.judges import FileContext, RubricJudge

RubricJudge(
    rubric=[
        {"criterion": "The report names the root cause of the crash.", "weight": 0.5},
        {"criterion": "The report proposes a fix.", "weight": 0.5},
    ],
    context=[FileContext([str(self.saved_submissions[0])])],
)
```

Each criterion is one LLM call that answers YES or NO with a short reason.
The reasons go into the metadata.

The `context` providers decide what the LLM sees.
Their output is joined with blank lines:

| Provider                         | Renders                                                                                                    |
| -------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| `FileContext(paths)`             | The content of each file. A file that can't be read shows up as an error line.                             |
| `TranscriptContext()`            | Every message, as `[role]: content`. `last=n` keeps the last `n`.                                          |
| `TranscriptContext(tool="bash")` | Calls to one tool with their arguments and results; `tool="*"` for all tools. `last=n` keeps the last `n`. |

Subclass `RubricContext` and implement `render(transcript) -> str` for your own provider.

### AlwaysPassJudge

`AlwaysPassJudge()` scores 1 and continues.
Use it for steps that only set something up, or as a placeholder.

## Combining judges

Combine judges with `&` and `|`.
They behave like Python's `and` and `or`, judging pass or fail by `continue_task`:

- `a & b` runs the judges in order and stops at the first that fails.
  It returns that `Scoring`, or the last one if all pass.
- `a | b` stops at the first that passes.
  It returns that `Scoring`, or the last one if all fail.

```python
RegexJudge([re.compile(r"answer: 42")]) & ExecutableJudge([...])
```

The combined `Scoring` keeps the score and `continue_task` of the judge it returns.

## Writing your own judge

Subclass `Judge` and implement `evaluate`:

```python
from typing import override

from karotte.judges import Judge
from karotte.schemas.scoring import Scoring
from karotte.schemas.transcript import Transcript


class FinalMessageJudge(Judge):
    """Passes when the student's last message contains `expected`."""

    def __init__(self, expected: str) -> None:
        self.expected = expected

    @override
    def evaluate(self, transcript: Transcript) -> Scoring:
        replies = [m for m in transcript.messages if m.role == "assistant" and m.content]
        last = str(replies[-1].content) if replies else ""
        passed = self.expected in last
        return Scoring(score=float(passed), metadata={"last_message": last}, continue_task=passed)
```

A judge can score from:

- `transcript.messages`: every message, including tool calls and tool results.
- `transcript.events`: the raw event log.
  The event types are in [`karotte.schemas.transcript`](https://github.com/preferencemodel/karotte/blob/main/src/karotte/schemas/transcript.py).
- Files the student wrote.
  Read the copies your `pre_scoring_hook` saved, not the originals; see [Saving submissions](#saving-submissions).

Custom judges compose with `&` and `|` like the built-in ones.

## Student misbehavior

Situations in which the student did something dubious (and potentially malicious) should score 0, not fail the run.
Examples include submitting a symlink to a root-only file or a FIFO that blocks that blocks indefinitely on read.

Raise `StudentMisbehaviorError` anywhere during scoring:

```python
from karotte import StudentMisbehaviorError

raise StudentMisbehaviorError(f"{path} is a symlink")
```

The step scores 0 with `continue_task=False` and your message in `metadata["misbehavior"]`.

!!! warning

    Karotte catches it only in `pre_scoring_hook`, the `judge` property and `judge.evaluate`.
    Raised from a tool, `pre_hook`, `post_hook` or anywhere else, it fails the run as an error.
    A scoring script run by `ExecutableJudge` is a separate process, so the exception never reaches Karotte.
    Your scoring script needs to handle student misbehavior and convert it to a score of 0 instead of crashing.

## Saving submissions

There is a crucial moment during a run when the student is done with its work and Karotte needs to grade it.
You should assume that the student is not trustworthy and will try to break the judge if possible.

Karotte's `default` template goes to great lengths to get the environment into a state where you can be sure that the student can't interfere with your grading anymore.
You only need to set `submission_paths` to point to the files you need from the student for grading:

```python
import sys
from pathlib import Path

from karotte import Step
from karotte.judges import ExecutableJudge

from environment.paths import STUDENT_DATA_DIR
from environment.submissions import collect_submission


class MyStep(Step):
    saved_submissions: tuple[Path, ...] = ()

    @property
    def submission_paths(self) -> tuple[Path, ...]:
        return (STUDENT_DATA_DIR / "server.rs",)

    @property
    def instructions(self) -> str:
        return f"Write the server to {self.submission_paths[0]}."

    def pre_scoring_hook(self):
        self.saved_submissions = collect_submission(self.config, self.submission_paths)

    @property
    def judge(self):
        assert self.saved_submissions, "pre_scoring_hook has not run"
        return ExecutableJudge(
            [
                sys.executable,
                "-m",
                "environment.tasks.my_task.scoring_script",
                str(self.saved_submissions[0]),
                "score_output.txt",
            ]
        )
```

The judge must use the copy in `saved_submissions`, never the path the student wrote to.
The scoring script reads the submission from its first argument and has to handle a path that doesn't exist.

### collect_submission

`collect_submission(config, paths, save_submission_kwargs=None)` lives in `src/environment/submissions.py` and returns the copies in the order of `paths`.
It takes the following steps, in order:

1. `kill_processes(STUDENT_UID)`, so no lingering student process can mess with the scoring.
2. `delete_files(STUDENT_UID, extend_exclude=paths)`, which removes everything else the student owns, so there is room for the copy.
3. `save_submission(path, **save_submission_kwargs)` for each path.
4. `delete_files(STUDENT_UID)`, which removes the originals once the copies are safe.
5. `save_artifact` on each copy (see [Artifacts and transcripts](artifacts-and-transcripts.md)).

`kill_processes` and `delete_files` are described in [Student resources](../running/student-resources.md).
Call the functions yourself if you need a different order.

!!! warning

    `collect_submission` deletes all files the student owns.
    In a task with several steps, a later step starts without the student's earlier files unless you put them back.

### save_submission

```python
from karotte import save_submission

copy = save_submission(source)
```

`save_submission` copies a student's submission to a safe place.
By default it creates a fresh directory under `~/.config/karotte/submissions/`, but you can override it by passing a `dest` argument.

Anything that looks like tampering raises `StudentMisbehaviorError`, for example:

- a symlink,
- a special file,
- a missing parent directory,
- a destination error the student can provoke (full disk, name collision, over-long path),
- or a file tree that is too large, has too many entries or is nested too deeply.

[`save_submission.py`](https://github.com/preferencemodel/karotte/blob/main/src/karotte/save_submission.py) has the limits and their defaults.
Pass the limits through `collect_submission` as `save_submission_kwargs`.
