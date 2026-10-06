# Installation

## Requirements

- Python 3.12 or newer.
- [uv](https://docs.astral.sh/uv/).
- A runtime to run environments in.
  See [below](#runtime-prerequisites).

## Install Karotte

```sh
uv tool install karotte
karotte --version
karotte --help
```

This installs the `karotte` command you use to create environments.

## Runtime prerequisites

`karotte run` and `karotte build` pick a runtime based on your system or you can select one via the `--runtime` option.

| Platform                                | Default runtime   | What you need                                                                                                           |
| --------------------------------------- | ----------------- | ----------------------------------------------------------------------------------------------------------------------- |
| macOS 26+ on Apple silicon              | `apple-container` | Apple [`container`](https://github.com/apple/container/releases) 1.4.1 or newer, started with `container system start`. |
| Linux x86_64 or aarch64 with `/dev/kvm` | `firecracker`     | Membership in the `kvm` group, docker with buildx (it builds the image), and `pasta` (package `passt`).                 |
| Everything else                         | `docker`          | docker with [buildx](https://github.com/docker/buildx#installing).                                                      |

You can always explicitly use docker with `--runtime docker`, or podman with `--runtime podman`, instead.
Ubuntu's `docker.io` package lacks buildx; install `docker-buildx` too.

If the default VM runtime isn't set up, Karotte stops before the run and tells you what's missing.

See [Runtimes](../running/runtimes.md) for the full setup of each runtime and what it gives the student.

## Next steps

- [Quick start](quick-start.md): create an environment and run its example task.
- [Concepts](concepts.md): what environments, tasks, steps and judges are.
