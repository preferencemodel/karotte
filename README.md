# karotte

karotte runs LLM agents on tasks and scores the results.

You write an *environment*: a Python project with one or more tasks. A task is a
list of steps. Each step gives the agent instructions and a judge that decides
whether the agent succeeded. karotte builds the environment into a container
image, lets the model work inside it through tools like `bash`, and records every
message, tool call and score in a transcript.

The agent runs as an unprivileged user with its own resource limits, a firewall,
and a disk quota, so a task can hand it a real shell without trusting it.
The model under test is called the *student*.

## Install

karotte needs Python 3.12+ and [uv](https://docs.astral.sh/uv/). Runs go into a
VM by default: Apple `container` on macOS, Firecracker on Linux (see
[Runtimes](#runtimes)). docker with
[buildx](https://github.com/docker/buildx#installing) (or podman with
`--runtime podman`) works too, and is what tasks needing devices passed through
use. Ubuntu's `docker.io` lacks buildx; install `docker-buildx` too. The justfiles need
[just](https://github.com/casey/just) 1.40 or newer; `uv sync --extra dev`
installs one into the venv.

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
uv run karotte create-run-config --model claude-fable-5
export ANTHROPIC_API_KEY=...
uv run karotte run --config run_config.json
```

Commands that load your tasks (`create-run-config`, `run`, `tasks list`,
`check`) have to run in the environment's venv, hence `uv run`.
`setup_data.py` prepares data the image needs, such as model weights; the
`default` template's version does nothing. `karotte check` loads every task and
tool and fails if one doesn't load; the image build runs it too.

`karotte run` builds the image, runs the task, and writes the transcript to
`out/transcript.json`. Files the task saves as artifacts go to
`out/<run_id>_artifacts/`. `karotte dashboard out/` shows the transcripts in `out/`.
`karotte models list` shows which models karotte knows; model ids are passed to
[litellm](https://docs.litellm.ai/). The API key is `model_api_key` in the run
config. Without one, karotte reads it from the key variable of the model's
provider, such as `ANTHROPIC_API_KEY` or `OPENAI_API_KEY`, when the run starts.
`--model-api-key` sets a key, or a `$VAR` that names another variable. A
`RubricJudge` whose task names no model uses `rubric_judge_model`, by default
the run's own model and key. A set `rubric_judge_model` reads its key,
`rubric_judge_api_key`, the same way the run's model does, but a missing one
doesn't stop the run from starting. A judge that gets no verdict from its model
fails the run instead of scoring 0. With the fake model or a `pt/` model, set
`rubric_judge_model` or name a model in the task.

Useful while you work on a task:

- `uv run karotte run --config run_config.json --dev` skips the image build and
  mounts your `src/environment/` into the container.
- `uv run karotte tasks list` shows the tasks the environment defines;
  `create-run-config --task <task-id>` picks one.

### The container image

Every `karotte run` without `--dev` rebuilds the image first. podman and docker
reuse cached layers, so an unchanged environment builds quickly.

`--dev` skips the build and runs the existing image, with your
`src/environment/` mounted over the installed copy. Rebuild after changing
dependencies, the `Containerfile` or data.

`karotte build` builds the same image without running a task. `run` always uses
the image tagged `karotte`, which is `build`'s default tag, and has no option to
pick another. `build --tag` is for images you use elsewhere.

### Model endpoints and proxies

`karotte run --proxy <url>` sends the model calls to `<url>` instead of the
provider. Pass the root URL, without `/v1`:

- Claude models go to `<url>/v1/messages`. karotte sets `ANTHROPIC_BASE_URL` to
  `<url>` in the container, and litellm adds `/v1/messages`.
- Other models go to `<url>/v1`, so the endpoint needs an OpenAI-compatible API
  there. `vertex_ai/` models ignore the proxy.

The same flag points karotte at any endpoint that serves these paths. The URL is
used inside the container, where `localhost` is the container itself. Variables
like `ANTHROPIC_BASE_URL` in your shell don't reach the container. With
`--no-containerized`, karotte sets `ANTHROPIC_BASE_URL` and `KAROTTE_PROXY_URL`
in its own process instead, unless one of them is already set.
`--no-containerized` only runs inside a karotte container image, where
`KAROTTE_CONTAINERIZED` is set.

A key is needed only if the endpoint asks for one. karotte still sends one; if
the variable it reads the key from is unset, `--proxy` fills in a placeholder.

Without `--proxy`, karotte uses the URL a plugin registers under
`karotte.default_proxy_url` (see [Plugins](#plugins)). Without such a plugin,
calls go straight to the provider. `--no-proxy` ignores the plugin's URL.

`KAROTTE_INFERENCE_SERVICE_TIER=priority` asks Fireworks and Vertex AI Gemini
for their priority tier, `auto` only once the provider runs out of capacity.
Unset, the default, uses the provider's default tier.

## Runtimes

`karotte run` and `karotte build` pick the runtime with `--runtime`. Without it
they use the platform's VM, so the agent gets a kernel of its own:

| Platform | Default |
| --- | --- |
| macOS 26+ on Apple silicon | `apple-container` (Apple `container` 1.4.1+) |
| Linux x86_64 or aarch64 with `/dev/kvm` | `firecracker` |
| Hardware a plugin marks `passthrough`, an Intel Mac or one before macOS 26, Linux without `/dev/kvm`, other platforms | `docker` |

A VM runtime the machine can run but that isn't set up stops before the run
with what's missing and how to fix it. It never falls back to a container on its own; pass
`--runtime docker` (or `podman`) for that.

Setup:

- `apple-container`: install Apple's
  [`container`](https://github.com/apple/container/releases) and run
  `container system start`. Build contexts must not be under `/tmp`.
- `firecracker`: be in the `kvm` group, have docker with buildx (it builds the
  image) and `pasta` (package `passt`). The first run downloads a pinned
  Firecracker and guest kernel into `~/.cache/karotte`. pasta gives the guest
  its network without root. On Ubuntu 24.04, AppArmor keeps pasta from making
  the user namespace it needs until you lift the restriction:
  `sudo sysctl -w kernel.apparmor_restrict_unprivileged_userns=0` (until
  reboot). A model proxy on the same host isn't reachable from the guest; one
  elsewhere is. `KAROTTE_FIRECRACKER_JAILER=1`, with `karotte run` under sudo,
  starts the VM through Firecracker's jailer, which needs no such change.
- `docker`, `podman`, `docker:gvisor`, `nerdctl`: containers on the host kernel
  (gVisor's for `docker:gvisor`).

A VM gets the CPUs, memory and disk that `karotte.hardware_limits` gives the
task's hardware, or 2 CPUs and 4 GiB for the agent when no plugin answers. The
VM holds 1 GiB more than the agent's limit for the harness.

What each gives the agent by default:

| | `apple-container`, `firecracker` | `docker`/`podman` (runc) | `docker:gvisor` |
| --- | --- | --- | --- |
| Memory and process limits | enforced by a cgroup | watched and reaped | watched and reaped |
| Disk bytes and files | loop-mounted quota | watched and reaped | watched and reaped |
| Network | firewall, checked at start; `firecracker` also filters on the host | firewall, checked at start | network namespace per session |
| Fork-and-die process chains | PID namespace per session | swept | PID namespace per session |

`karotte check confinement`, run as root inside the sandbox, prints what yours
actually does and exits 1 when it gives the agent less than it should.

Environment variables a launcher or task may set:

| Variable | Effect |
| --- | --- |
| `KAROTTE_SANDBOX` | `runc`, `gvisor` or `vm`: what the sandbox is, when karotte can't tell. |
| `KAROTTE_STUDENT_NETWORK` | Unset or `strict`: the agent reaches localhost, the sandbox's own addresses and the model proxy. `internal`: also link-local and private ranges. |
| `KAROTTE_DISK_BUDGET_BYTES` | Cap on the agent's disk quota, chosen where the host's free space is known. VM launchers set it. |
| `KAROTTE_SANDBOX_MEMORY_BYTES` | Memory the sandbox holds for the agent, when no plugin says. VM launchers set it. |
| `KAROTTE_VM_LAUNCHER` | Set by karotte's VM runtimes. `karotte check confinement` then fails a VM without 1 GiB above the sandbox's memory; in a VM another launcher sized, it only warns. |
| `KAROTTE_FIRECRACKER_NETWORK` | `pasta` (the default) or `none`, for a VM without a network. |

## Writing tasks

Each task is a package in `src/environment/tasks/`: a directory whose
`__init__.py` defines the `Task` class. Plain files and directories starting with
`_` are skipped. `uv run just create-task <task-id>` copies the task template
(`just` comes with `uv sync --extra dev` and lives in the venv). A minimal task:

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

Judges in `karotte.judges`: `RegexJudge` matches the final message,
`ExecutableJudge` runs a scoring script, `RubricJudge` asks an LLM to grade
against a rubric, and `Judge` is the base class for your own. The example task
in the `default` template shows submissions, hints, and hooks that run before
scoring.

## Templates

A template is a directory of files, rendered with
[Jinja](https://jinja.palletsprojects.com/) into a new environment. karotte ships
two:

- `default`: a CPU environment with an example task. Every other template builds
  on it.
- `language-toolchains`: gives the agent exactly one language toolchain per task.

`karotte templates list` shows every installed template. Templates stack:
`karotte create-env my_env --template default --template language-toolchains`
renders `default` and then `language-toolchains` on top. A later template can
replace a file or override a Jinja block in it:

```
{% extends "default/CLAUDE.md" %}
{% block claude_md -%}
New content
{% endblock %}
```

`karotte update` brings an existing environment up to the latest templates with
a 3-way merge, so your own edits survive.

### Writing a template package

Templates can live in their own Python package. Put the template directories
(each with a `template.toml`) under one directory and register it under the
`karotte.templates` entry point:

```toml
[project.entry-points."karotte.templates"]
my_templates = "my_package:TEMPLATES_DIR"
```

`TEMPLATES_DIR` is a path to that directory. A `template.toml` holds the fields of
[`EnvironmentTemplate`](https://github.com/preferencemodel/karotte/blob/main/src/karotte/schemas/environment_template.py), for
example:

```toml
description = "Adds a Rust toolchain."
requires = ["default"]
```

Install the package next to karotte (`uvx --with my-package karotte create-env
...`). karotte records it in the environment's manifest, so `karotte update`
pulls it in again.

## Plugins

Besides templates, a package can extend karotte through these entry points:

| Entry point | Points at | Effect |
| --- | --- | --- |
| `karotte.cli` | a Typer app | Adds a subcommand named after the entry point. |
| `karotte.run_config_preprocessors` | `f(config) -> config` | Rewrites the run config before a run, in entry point name order. |
| `karotte.default_proxy_url` | a string | Default for `karotte run --proxy`. |
| `karotte.harness_secret_env` | a list of names | Environment variables hidden from the agent. |
| `karotte.platform_tooling_dirs` | a list of paths | Directories where a platform mounts its own tooling into every container; hidden from graded toolchain runs. |
| `karotte.age_delay_exemptions` | a list of package names | More packages exempt from uv's `exclude-newer` delay (karotte always is). |
| `karotte.update_migrations` | an object with `prepare`, `tool` and `migrate` | Moves envs made by an older release to the current names during `karotte update`. |
| `karotte.default_hardware` | a string | `required_hardware` of tasks that set none. karotte itself knows no hardware names. |
| `karotte.hardware_limits` | `f(hardware) -> HardwareLimits \| None` | Memory, disk and CPUs a sandbox on that hardware holds; `passthrough=True` marks hardware a VM can't hold, which runs under docker. Without it, the agent's memory limit is `KAROTTE_SANDBOX_MEMORY_BYTES`, else the sandbox's cgroup limit or RAM, less 1 GiB for the harness, and none where cgroups don't work. |
| `karotte.container_run_args` | `f(task, runtime) -> list[str]` | Extra arguments for the container engine's `run` (docker, podman or nerdctl), e.g. to pass devices through, in entry point name order. Raising refuses the launch with the exception's message. |

A plugin that fails to load is skipped with a warning, except a run config
preprocessor: that one fails the run.

## Connecting a backend

Without a backend, karotte writes the transcript to a file. To collect runs
centrally, set `backend_uri` in the run config. karotte then calls these HTTP
endpoints on it, with `run_id` as a query parameter (presign has it in the body):

| Request | Purpose |
| --- | --- |
| `POST /api/internal/create_transcript` | Start a transcript. |
| `GET /api/internal/transcript_length` | Number of stored events, 404 if none. |
| `POST /api/internal/append_transcript` | Body `{"event": ..., "seq": n}`. Writing the same `seq` again must overwrite, not append. |
| `POST /api/internal/update_run_state` | Body with `status` (`running`, `passed`, `failed`, `error`), `score`, and token counts. |
| `POST /api/artifacts/presign` | Body `{"run_id", "artifact_paths"}`, returns `{"presigned_urls": {path: url}}`. karotte PUTs each file gzipped. |

With the `external` agent the backend also supplies the model's messages:
`GET /api/internal/get_message` returns the next message (404 while there is
none) and `POST /api/internal/delete_message` consumes it.

Requests carry `Authorization: Bearer <token>` if the file at
`KAROTTE_BACKEND_TOKEN_PATH` (default `/var/run/secrets/service-account-token`)
exists. The events are the models in
[`karotte.schemas.transcript`](https://github.com/preferencemodel/karotte/blob/main/src/karotte/schemas/transcript.py).

## Development

```
just lint
just test
just test-template default
```

Every merge to `main` is released. The patch version is the commit count.

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
off by default, is under Modular's proprietary license and has its own terms even
if you never share the image. The rest only matters if you redistribute a built
image, for example by pushing it to a public registry.
