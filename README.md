# Karotte 🥕

Karotte is an open-source framework for building robust RL environments, made by
[Preference Model](https://preferencemodel.com).

**Documentation: [karotte.dev](https://karotte.dev)**

## Install

Karotte needs Python 3.12+ and [uv](https://docs.astral.sh/uv/). Runs go into a
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
export ANTHROPIC_API_KEY=...
uv run karotte run --task example-task --model anthropic/claude-fable-5
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
[Developing Karotte](https://karotte.dev/extending/developing-karotte/).

## License

Karotte is under the [MIT license](https://github.com/preferencemodel/karotte/blob/main/LICENSE). The templates in
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
