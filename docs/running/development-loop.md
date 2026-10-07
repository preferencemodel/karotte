# Development loop

## Skip the build with `--dev`

Every `karotte run` without `--dev` rebuilds the image first.
podman and docker reuse cached layers, so an unchanged environment builds quickly.
A changed one can take a while.

```sh
uv run karotte run --config run_config.json --dev
```

`--dev` skips the build and runs the existing image.
It bind-mounts your `src/environment/` over the installed copy at `/root/.venv/lib/python3.12/site-packages/environment/`.
That way, changes to task code, judges and tools take effect on the next run.

Build the image once before your first `--dev` run.
Rebuild it after you change any of these:

- dependencies (`pyproject.toml`, `uv.lock`, the venvs in `venvs/`), including a Karotte update
- the `Containerfile`
- data in `student_data/`, `shared_data/`, `root_data/` or `intermediate_data/`
- the CLI agents in the environment's manifest

You can combine `--dev` with `--mount`.

## Build without running

```sh
uv run karotte build
```

`karotte build` builds the same image without running a task.
`run` always uses the image tagged `karotte`, which is the default tag for `build`.
There's no option to make `run` use a different image.
Use `build --tag` for images you want to use elsewhere.

`--build-secret name=path` passes in a file that the `Containerfile` can mount with `RUN --mount=type=secret,id=<name>`.
`run` takes the same option.

## Mount host paths

`--mount` bind-mounts a host path into the container.
The format is `source:target[:ro]`.
Mounts are read-write by default.
Append `:ro` to make one read-only.
The target must be an absolute path, and the source must exist.

```sh
uv run karotte run --config run_config.json --mount /path/on/host:/path/in/container
uv run karotte run --config run_config.json --mount /data:/data:ro
```

To add several mounts, pass `--mount` more than once, or read them from a file with `@`:

```sh
uv run karotte run --config run_config.json --mount @mounts.txt
```

The file has one spec per line.
Karotte ignores blank lines and lines starting with `#`.

!!! warning

    With `-n` (parallel runs), all containers share the same bind mounts.
    That makes a read-write mount shared state between them, and Karotte warns you about it.
    Use `:ro` where you can.

The VM runtimes handle mounts differently.
`firecracker` supports read-only mounts only.
`apple-container` copies each mount in when the run starts, and copies the read-write ones back when it ends.
See [Runtimes](runtimes.md).

To change a task's behavior without a rebuild or a mount, pass values through `extra_config` in the run config.
See [Run config](run-config.md).

## Keep containers

`--keep-containers` keeps the containers after the runs instead of removing them, so you can inspect them or copy data out.
They're named `karotte_run_<run_id>`.
When the next run starts, Karotte removes leftover containers from earlier runs whose names start with `karotte_run_` followed by the longest common prefix of the new run ids.

When the run ends, Karotte prints how to copy the student's workdir out, for example:

```sh
docker cp karotte_run_<run_id>:/workdir/ ./out/
```

`firecracker` doesn't use a container.
Instead, Karotte keeps the VM's drives and logs where they are.

## Credentials in the image

`karotte check` runs as the last step of the image build.
It runs inside the image and fails the build if the image contains credential material.
It looks in the home directories (`/root`, the student's workdir, `/home/*`) for the places where credentials end up: uv's credential stores, `~/.netrc`, `~/.git-credentials`, the GitHub CLI's token, gcloud's credential files, `~/.aws/credentials`, `~/.kube/config`, Hugging Face tokens, keyring stores and SSH private keys.
Config files such as `.npmrc`, `.pypirc`, `.docker/config.json`, `pip.conf` and `uv.toml` fail the check only if they hold a secret inline.
If they read the secret from an environment variable, they pass.

A build secret mounted with `--mount=type=secret` stays out of the layers.
But a tool you give the secret to can still save something it derives from it, usually a cached token under the home directory.
That write ends up in a layer like any other file.
Making the file root-only doesn't help, because anyone who can pull the image can read it.

If the check fires, delete the file in the same `RUN` step that created it, and rotate the credential it came from.
It has to be the same step.
The check sees only the final filesystem.
If a credential is written in one layer and deleted in a later one, it passes the check but still ships in the layer that holds it.

## `--no-containerized`

`--no-containerized` runs the task in the current process instead of starting a container.
It only works inside a Karotte image, where the `Containerfile` sets `KAROTTE_CONTAINERIZED`.
Anywhere else, `karotte run` refuses it.
Karotte uses it inside the container it starts.

`--dev`, `--mount`, `--keep-containers` and `-n` greater than 1 all need a container, so you can't combine them with `--no-containerized`.

## Lint and test

An environment's `justfile` has these recipes:

| Recipe      | What it does                                                                           |
| ----------- | -------------------------------------------------------------------------------------- |
| `just lint` | Runs `ruff format --check`, `ruff check`, and a check of the uv supply-chain settings. |
| `just fix`  | Runs `ruff check --fix`.                                                               |
| `just fmt`  | Runs `ruff format`.                                                                    |
| `just test` | Runs `pytest`.                                                                         |

`uv sync --extra dev` installs `just` into the venv, so run the recipes through uv:

```sh
uv run just lint
uv run just test
```

`uv run just --list` shows every recipe, including the ones that create tasks.
The `default` template ships tests in `tests/` for task discovery and its other helpers.
Add your own tests there as you change things.

## Past transcripts

`uv run karotte dashboard <dir>` shows the transcripts in a directory.
By default, runs write to `out/`.
See [Artifacts and transcripts](../tasks/artifacts-and-transcripts.md).
