# Run config

A run config describes how to execute a run: the task, the model, its API key, and options such as limits.
`karotte run --config` accepts a path to a JSON file or the JSON itself:

```sh
uv run karotte run --config run_config.json
```

The JSON gets parsed as an [`EvaluationRunConfig`](https://github.com/preferencemodel/karotte/blob/main/src/karotte/schemas/evaluation_run_config.py).

## Flags

Without `--config`, `karotte run` builds the config from its flags:

```sh
uv run karotte run --task find-python --model anthropic/claude-fable-5
```

`--task` and `--model` are then required.
The run gets a random `run_id`, and `transcript_file` defaults to `out/transcript.json`.

| Flag                 | Field              |
| -------------------- | ------------------ |
| `--task`             | `task_id`          |
| `--model`            | `model`            |
| `--model-api-key`    | `model_api_key`    |
| `--reasoning-effort` | `reasoning_effort` |
| `--transcript-file`  | `transcript_file`  |

With `--config`, the flags override the config's fields, so one config can be run against several models:

```sh
uv run karotte run --config run_config.json --model openai/gpt-5.5
```

The other fields have no flag; set them in a config.

## Create a run config

```sh
uv run karotte create-run-config --model anthropic/claude-fable-5 --task find-python
```

It loads your tasks, so run it from the environment repo.
It writes a random `run_id` and sets `transcript_file` to `out/transcript.json`:

TODO: Can the following be generated during docs build?

```json
{
    "run_id": "3f9c1a2b",
    "task_id": "example-task",
    "agent": null,
    "model": "anthropic/claude-fable-5",
    "model_api_key": null,
    "rubric_judge_model": null,
    "rubric_judge_api_key": null,
    "use_hints": true,
    "reasoning_effort": null,
    "turn_limit": null,
    "step_time_limit_seconds": null,
    "on_step_time_limit": "error",
    "inject_time_remaining_counter": true,
    "step_context_window_limit": null,
    "on_step_context_window_limit": "error",
    "inject_context_remaining_counter": true,
    "mcp_server_config": {
        "host": "0.0.0.0",
        "port": 8080,
        "profile_tool_calls": false
    },
    "websocket_config": {
        "host": "0.0.0.0",
        "port": 8001
    },
    "transcript_file": "out/transcript.json",
    "use_fake_model": false,
    "extra_config": null,
    "backend_uri": null,
    "save_artifacts": true
}
```

If you rename a task or want to run another one, change `task_id`.

## Extra config

`extra_config` is a free-form dict passed to your task unchanged; see [Tasks and steps](../tasks/tasks-and-steps.md#extra-config) for how a task reads it.

```json
{
    "extra_config": {
        "difficulty": "hard",
        "extra_system_prompt": "Explain each command before you run it."
    }
}
```

These keys have a built-in meaning:

| Key                          | Value                     | Effect                                                                                                                                                           |
| ---------------------------- | ------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `extra_system_prompt`        | string                    | Appended to the system prompt after a blank line.                                                                                                                |
| `system_prompt_override`     | string                    | Replaces the system prompt. Wins over `extra_system_prompt`.                                                                                                     |
| `extra_task_instructions`    | string or list of strings | Appended to the step's instructions after a blank line. A string applies to every step; a list has one entry per step, and steps past its end are unchanged.     |
| `task_instructions_override` | string or list of strings | Replaces the step's instructions. Same string-or-list rules. Wins over `extra_task_instructions`.                                                                |
| `extra_artifact_paths`       | string or list of strings | Absolute paths saved as [artifacts](../tasks/artifacts-and-transcripts.md) after each step, before `pre_scoring_hook`. Missing paths are skipped with a warning. |

The first two are read by the `default` template's `get_system_prompt()`, so they only work in tasks that use it.
Karotte itself applies the latter three.

## Models

`model` is a [litellm](https://docs.litellm.ai/docs/providers) model id with its provider prefix, such as `anthropic/claude-fable-5`, `openai/gpt-5` or `together_ai/moonshotai/Kimi-K3`.

`karotte models list` shows the models Karotte knows by name, each with its output-token ceiling and the reasoning effort levels it accepts.
`--json` prints the catalog as JSON.

```sh
karotte models list
```

It also lists prefixes under which any model id is accepted, such as `together_ai/` and `fireworks_ai/`.
A model Karotte doesn't know still runs, with a 64k output-token ceiling and no reasoning effort parameter.

## API keys

Without `model_api_key`, Karotte reads the key from the model provider's variable in your shell when the run starts:

| Model                       | Variable                                                 |
| --------------------------- | -------------------------------------------------------- |
| `anthropic/...`, `claude-*` | `ANTHROPIC_API_KEY`                                      |
| `openai/...`                | `OPENAI_API_KEY`                                         |
| `gemini/...`                | `GEMINI_API_KEY`                                         |
| `mistral/...`               | `MISTRAL_API_KEY`                                        |
| `together_ai/...`           | `TOGETHERAI_API_KEY`                                     |
| other `<provider>/...`      | `<PROVIDER>_API_KEY`, e.g. `XAI_API_KEY`, with a warning |

To use another key, set `model_api_key` to the key itself or to a `$VAR` reference to another variable; `karotte run --model-api-key` and `create-run-config --model-api-key` set it for you.
If the variable is unset, the run stops with an error, unless it uses the fake model or `--prepare-only`.

### Rubric judge

By default, `RubricJudge` grades with the run's own model and API key.
To grade with a different model, set `rubric_judge_model`.
Karotte then reads its key from `rubric_judge_api_key`, or from the provider's variable in the table above.
If that key is missing, the run still starts.

A model passed to `RubricJudge(model=...)` in the task overrides both.

The fake model and `pt/` models can't grade.
For those runs, set `rubric_judge_model` or pass a model to `RubricJudge` in the task.

## Reasoning effort

`reasoning_effort` takes:

- `"min"` or `"max"`: the model's lowest or highest level, whatever the provider calls it
- one of the provider's own level names, such as `"medium"`, if the model accepts it
- `null`: send nothing and use the provider's default

Providers don't agree on what the levels are called or how many there are, so Karotte automatically selects the right level if you pass `min` or `max`.
`karotte models list` shows available levels per model.

## Limits

Use these fields to limit how much work the student can do:

| Field                              | Type                                | Meaning                                                                                                                                                      |
| ---------------------------------- | ----------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `turn_limit`                       | `int` or `null`                     | Maximum number of model responses in the run. When it's reached, the run ends with an error.                                                                 |
| `step_time_limit_seconds`          | `float`, list of `float`, or `null` | Maximum wall-clock time per step in seconds, counted from when the step's instructions are sent.                                                             |
| `on_step_time_limit`               | `"error"` or `"score"`              | What happens when a step runs out of time. `"error"` (the default) ends the run with an error. `"score"` ends the step and scores the student's work so far. |
| `inject_time_remaining_counter`    | `bool`                              | Tell the student how much time is left. Defaults to `true`.                                                                                                  |
| `step_context_window_limit`        | `int`, list of `int`, or `null`     | Maximum input tokens per turn in a step, based on what the provider reported for the previous turn.                                                          |
| `on_step_context_window_limit`     | `"error"` or `"score"`              | Same as `on_step_time_limit`, but for the context limit.                                                                                                     |
| `inject_context_remaining_counter` | `bool`                              | Tell the student how much context is left. Defaults to `true`.                                                                                               |

The time and context limits apply to each step separately.
Pass a single value to use it for every step, or a list with one value per step.
Steps beyond the end of the list have no limit.
For example, `step_time_limit_seconds: 600` gives the student ten minutes for each step of a two-step task.

Karotte checks both step limits between turns.
It doesn't interrupt a turn in progress, so a step can go over its limit by up to one turn.
The context limit uses the input-token count the provider reports for each turn.
It measures how large the student's context is, not how many tokens were billed over the whole step.
If the provider doesn't report usage, the context limit never triggers.

When a counter is on, Karotte adds a note like `Time remaining: 540 seconds` or `Context remaining: 12000` to the last tool result of each turn, so the model can pace itself.
When it's off, the student hits the limit without warning.
A counter has no effect unless its limit is set.

### Limit support by agent

Agents that run a step in their own process can only enforce the limits their CLI supports.

| Limit                              | `builtin`                          | `external` | CLI agents                                |
| ---------------------------------- | ---------------------------------- | ---------- | ----------------------------------------- |
| `turn_limit`                       | Yes                                | Yes        | Per step, through the CLI's `--max-turns` |
| `step_time_limit_seconds`          | Yes                                | Yes        | Yes, by killing the CLI's processes       |
| `inject_time_remaining_counter`    | Yes                                | Yes        | No                                        |
| `step_context_window_limit`        | Yes, if the provider reports usage | No         | No                                        |
| `inject_context_remaining_counter` | Yes, if the provider reports usage | No[^ext]   | No                                        |

[^ext]: The external agent doesn't report token usage, so its context counter, if on, always shows the full limit.

## Agents

The agent produces the student's messages and tool calls for each step.
Don't confuse it with the model: `model` picks which LLM answers, `agent` picks the harness that calls it.
Whichever agent you use, Karotte still runs the task's hooks, judges and artifacts.

| Agent      | What it does                                                                                                                                   |
| ---------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| `builtin`  | The default. Karotte calls the model through litellm, streams its responses, retries failed calls and runs the tool calls.                     |
| `external` | A backend sends each of the model's messages, and Karotte runs the tool calls. Needs `backend_uri`; see [Backend](../extending/backend.md).    |
| CLI agents | A coding agent's own CLI, installed in the image, works on each step as the student and calls the tools itself. `mistral-vibe` is one example. |

Set `agent` in the run config:

```json
{
    "agent": "mistral-vibe",
    "model": "mistral/<model-id>",
    "model_api_key": "$MISTRAL_API_KEY"
}
```

`karotte agents list` shows the agents your Karotte version supports, and the version it pins for each CLI agent.
`--json` prints the list as JSON.

```sh
karotte agents list
```

### CLI agents

`builtin` and `external` work with any image.
A CLI agent also has to be installed in the image at build time, so setting `agent` in the run config isn't enough.
Add it when you create the environment:

```sh
karotte create-env my_env --agent mistral-vibe
```

or add it to an existing environment and rebuild:

```sh
uv run karotte agents add mistral-vibe
uv run karotte build
```

Both commands record the agent in the environment's `.manifest.json`.
During the image build, `karotte agents install` installs the pinned version; it only works inside a build.
`karotte agents remove <name>` removes an agent from the manifest, and the next build leaves it out.

CLI agents differ from `builtin` in a few ways:

- They run as the student and call the tool server directly.
  If they bring their own tools, such as a shell, those replace the task's tools of the same name.
- They send their own system prompt instead of the task's.
- `reasoning_effort` isn't applied.
- The student's firewall only lets them reach the model proxy, so sandboxed runs need `--proxy`.
  `mistral-vibe` sends its calls to `<proxy>/v1` as an OpenAI-compatible endpoint, or straight to Mistral's API when there's no proxy.

## Fake model

The fake model replays messages you write in advance, so you can test your tools, hooks and judges without waiting for or paying for a real model.
It needs no API key.
To use it, set `use_fake_model` to `true` and write a `get_messages` function in `src/environment/fake_model.py`.
The `default` template includes one:

```python
from karotte import EvaluationRunConfig
from karotte.schemas import Message


def get_messages(config: EvaluationRunConfig) -> list[Message]:
    return [
        Message(role="assistant", content="path: /workdir/.venv/bin/python"),
    ]
```

Each turn uses the next message in the list.
A message with text and no tool calls ends the step, so finish each step with one.
If the list runs out, the run fails.

`use_fake_model` takes the place of whatever agent the config names.

## Model endpoints and proxies

`karotte run --proxy <url>` sends model calls to `<url>` instead of to the provider.
Pass the base URL without `/v1`:

- Claude models go to `<url>/v1/messages`.
  Karotte sets `ANTHROPIC_BASE_URL` to `<url>` inside the container, and litellm adds `/v1/messages`.
- Other models go to `<url>/v1`, so the endpoint needs an OpenAI-compatible API there.

You can use the same flag to point Karotte at any endpoint that serves these paths.
The URL is used from inside the container, so `localhost` means the container, not your machine.
Variables like `ANTHROPIC_BASE_URL` set in your shell don't reach the container.

The endpoint may not require a key and instead inject its own key.
Karotte always sends a key, though, so if the key's variable isn't set, `--proxy` fills in a placeholder for both the run's model and the judge model.

If you don't pass `--proxy`, Karotte uses the URL a plugin registers under `karotte.default_proxy_url` (see [Plugins](../extending/plugins.md)).
Without such a plugin, calls go directly to the provider.
`--no-proxy` ignores the plugin's URL.
