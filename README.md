# karotte

karotte runs LLM agents on tasks and scores the results.

You write an _environment_: a Python project with one or more tasks. A task is a
list of steps. Each step gives the agent instructions and a judge that decides
whether the agent succeeded. karotte builds the environment into a container
image, lets the model work inside it through tools like `bash`, and records every
message, tool call and score in a transcript.

The agent runs as an unprivileged user with its own resource limits, a firewall,
and a disk quota, so a task can hand it a real shell without trusting it.
The model under test is called the _student_.

**Documentation: [karotte.dev](https://karotte.dev)**

## Install

karotte needs Python 3.12+ and [uv](https://docs.astral.sh/uv/). Runs go into a
VM by default: Apple `container` on macOS, Firecracker on Linux. docker or
podman work too. See [Installation](https://karotte.dev/getting-started/installation/)
and [Runtimes](https://karotte.dev/running/runtimes/) for what each needs.

```
uv tool install karotte
```

## Quick start

Create an environment from the `default` template and run its example task:

```
karotte create-env my_env
cd my_env
uv sync --extra dev
uv run setup_data.py
uv run karotte create-run-config --model anthropic/claude-fable-5
export ANTHROPIC_API_KEY=...
uv run karotte run --config run_config.json
```

`karotte run` builds the image, runs the task, and writes the transcript to
`out/transcript.json`. `karotte dashboard out/` shows it. The
[quick start](https://karotte.dev/getting-started/quick-start/) explains each
step, and [Writing tasks](https://karotte.dev/tasks/tasks-and-steps/) shows how
to add your own.

## Development

```
just lint
just test
just test-template default
```

Every merge to `main` is released. See
[Developing karotte](https://karotte.dev/extending/developing-karotte/).

## License

karotte is under the [MIT license](https://github.com/preferencemodel/karotte/blob/main/LICENSE). The templates in
`src/karotte/templates/` are under [MIT No Attribution](https://github.com/preferencemodel/karotte/blob/main/src/karotte/templates/LICENSE)
(`MIT-0`), so environments created from them need no license notice.

### Third-party software in built images

Images built from the templates contain third-party software under its own
licenses. They are based on Amazon Linux 2023. The `language-toolchains`
template installs GPL and LGPL software (gcc, GnuCOBOL, GNU Prolog, Free Pascal,
the libraries bundled with Julia, and others), Amazon Corretto (GPL-2.0 with the
Classpath Exception) and Clojure (EPL-1.0) for the languages you enable. Mojo,
off by default, has its own license terms; check them for the version you
enable. If you distribute a built image, you're responsible for complying with
the licenses of the software in it.
