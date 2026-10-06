# Quick start

This creates an environment from the `default` template and runs its example task.
You need Karotte and a runtime installed; see [Installation](installation.md).

## Create an environment

```sh
karotte create-env my_env
cd my_env
uv sync --extra dev
```

`create-env` renders the `default` template into `my_env/` and locks its dependencies.
`uv sync --extra dev` installs the environment, Karotte and the dev tools (ruff, pytest, just) into `.venv`.

## Run the task

`anthropic/claude-opus-5-5` needs `ANTHROPIC_API_KEY`:

```sh
export ANTHROPIC_API_KEY=...
uv run karotte run --task example-task --model anthropic/claude-opus-5-5
```

`karotte run` builds the container image, starts it in the runtime, and runs the task.
A terminal UI shows the run as it happens; `--no-ui` prints the output to the terminal instead.

`uv run karotte tasks list` shows all tasks in the environment.

`uv run karotte models list` shows the models Karotte knows.
Model ids are passed to [litellm](https://docs.litellm.ai/).

Karotte reads the model's API key from the provider's variable, such as `ANTHROPIC_API_KEY` or `OPENAI_API_KEY`, when the run starts.
`--model-api-key` sets another key, or a `$VAR` reference to another variable.

For options without a flag, write a run config; see [Run config](../running/run-config.md).

The example task has two steps: the student finds the path of its Python executable, then writes the Python version to a file.
Its code is in `src/environment/tasks/example_task/`.

## Look at the results

The transcript of a run goes to `out/transcript.json`.
Files the task saves as artifacts go to `out/<run_id>_artifacts/`.
See [Artifacts and transcripts](../tasks/artifacts-and-transcripts.md).

To browse the transcripts in `out/` later:

```sh
uv run karotte dashboard out/
```

## Next steps

- [Concepts](concepts.md) explains the parts of an environment and how a run flows.
- [Tasks and steps](../tasks/tasks-and-steps.md) shows how to write your own task.
- [Development loop](../running/development-loop.md) shows how to skip the image build while you work on a task.
