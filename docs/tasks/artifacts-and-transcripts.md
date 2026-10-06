# Artifacts and transcripts

Every run produces a transcript.
A task can also save files from the sandbox as artifacts, so you can inspect them after the run.

## Artifacts

Call `save_artifact()` from a hook to keep a file or directory:

```python
from pathlib import Path

from karotte import Step, save_artifact


class MyStep(Step):
    def post_hook(self) -> None:
        save_artifact(self.config, Path("/some/path/report.md"))
```

A directory is copied with everything in it.
A path that doesn't exist is skipped with a warning.
With `save_artifacts: false` in the run config, `save_artifact()` does nothing.

`post_hook` runs after the step is scored.
If your `pre_scoring_hook` or judge deletes student files, save them before that.
The `default` template's `collect_submission()` already saves the submissions it collects as artifacts.

!!! warning

    `save_artifact` doesn't validate what it copies.
    Only call it on files you have "taken into custody" via something like `collect_submission()`.
    See [Scoring](scoring.md).

### Where artifacts go

Without a backend, artifacts are copied into a `<run_id>_artifacts/` directory next to the transcript, so `out/<run_id>_artifacts/` with the default run config.
If that directory already exists, Karotte appends `_2`, `_3` and so on instead of overwriting it.
`karotte run` prints where the transcript and artifacts are when it finishes.

With `backend_uri` set in the run config, artifacts are uploaded to the backend instead.
See [Connecting a backend](../extending/backend.md).

### Artifacts without code changes

To capture extra files without touching the environment, list their absolute container paths under `extra_artifact_paths` in the run config's `extra_config`.
Karotte saves them after each step, before the `pre_scoring_hook` runs.
See [Run config](../running/run-config.md#extra-config).

## Transcripts

The transcript records every event of a run: messages, tool calls and their results, scores, token usage and errors.
Karotte writes it as JSON to the run config's `transcript_file`, which `karotte create-run-config` sets to `out/transcript.json`.
The file is written when the run ends.
A new run with the same config replaces it.
With `-n 3`, the runs write `transcript_0.json` to `transcript_2.json`.

The transcript holds the run id and a list of events.
Each event has a `timestamp` and a `type`.
The full schema is in [`karotte/schemas/transcript.py`](https://github.com/preferencemodel/karotte/blob/main/src/karotte/schemas/transcript.py).
Streamed message chunks go to live viewers only and aren't saved.

To read a transcript in Python, validate it with the `Transcript` model:

```python
from pathlib import Path

from karotte.schemas.transcript import Transcript

transcript = Transcript.model_validate_json(Path("out/transcript.json").read_text())
for message in transcript.messages:
    print(message.role, message.content)
```

## Viewing transcripts

`karotte dashboard` opens past transcripts in a terminal UI:

```bash
karotte dashboard out/
```
