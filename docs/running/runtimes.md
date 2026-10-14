# Runtimes

The runtime is what runs the environment's image.
It's either a VM with its own kernel, or a container that shares the host's kernel.
Pick one with `--runtime` on `karotte run` and `karotte build`:

| Runtime                       | What it is                                                                                               |
| ----------------------------- | -------------------------------------------------------------------------------------------------------- |
| `apple-container`             | A VM per run through Apple's [`container`](https://github.com/apple/container).                          |
| `firecracker`                 | A [Firecracker](https://firecracker-microvm.github.io/) microVM per run. The image is built with docker. |
| `docker`, `podman`, `nerdctl` | A container on the host kernel, under the engine's default OCI runtime (usually runc).                   |
| `docker:gvisor`               | A docker container under [gVisor](https://gvisor.dev/), which runs its own kernel in user space.         |
| `modal`                       | A [Modal](https://modal.com/) sandbox with its own kernel per run, in the cloud. Modal builds the image. |

## The default

If you don't pass `--runtime`, Karotte uses your platform's VM, so the student gets its own kernel:

| Platform                                                                                                              | Default                                      |
| --------------------------------------------------------------------------------------------------------------------- | -------------------------------------------- |
| macOS 26+ on Apple silicon                                                                                            | `apple-container` (Apple `container` 1.4.1+) |
| Linux x86_64 or aarch64 with `/dev/kvm`                                                                               | `firecracker`                                |
| Hardware a plugin marks `passthrough`, an Intel Mac or one before macOS 26, Linux without `/dev/kvm`, other platforms | `docker`                                     |

`karotte run` logs which runtime it picked.
If your machine can run the VM but it isn't set up yet, Karotte stops before the run and tells you what's missing and how to fix it.
It never switches to a container on its own.
Pass `--runtime docker` (or `podman`) if you want one.
It never picks `modal` either, which runs your environment on Modal's servers and bills your Modal account.

Tasks that need devices passed through, such as a GPU, run under docker.
A [plugin](../extending/plugins.md) marks that hardware `passthrough` in `karotte.hardware_limits` and adds the device flags through `karotte.container_run_args`.

## Setup

### apple-container

Install Apple's [`container`](https://github.com/apple/container/releases) (1.4.1 or later) and start its service:

```sh
container system start
```

Don't put the environment under `/tmp`, `/private/tmp` or any other path that goes through a symlink, such as `/var`.
Apple's builder copies the directories in it as empty directories, so Karotte refuses to build from there.

Images that only have an amd64 variant run under Rosetta.

`--mount` paths aren't shared live.
They're copied into the VM when the run starts, and read-write ones are copied back when it ends.

### firecracker

You need:

- read and write access to `/dev/kvm`
- docker with [buildx](https://github.com/docker/buildx#installing) to build the image
- `pasta` (package `passt`), `ip` and `iptables` for the guest's network
- `mkfs.ext4` and `debugfs` (package `e2fsprogs`) to build the guest's drives

To get access to `/dev/kvm`, add yourself to the `kvm` group and log in again:

```sh
sudo usermod -aG kvm $USER
```

On the first run, Karotte downloads Firecracker and a guest kernel from a [Kata Containers](https://katacontainers.io/) release into `~/.cache/karotte/firecracker` (under `$XDG_CACHE_HOME` if it's set).
Each Karotte release fixes which versions it downloads and their SHA-256 checksums, and stops with an error if a download doesn't match.

pasta gives the guest a network without needing root.
On Ubuntu 24.04, AppArmor stops pasta from creating the user namespace it needs.
To lift that restriction until the next reboot, run:

```sh
sudo sysctl -w kernel.apparmor_restrict_unprivileged_userns=0
```

Or set `KAROTTE_FIRECRACKER_JAILER=1` and run `karotte run` with sudo.
Karotte then starts the VM through Firecracker's jailer, which doesn't need that change.

The guest can't reach a model proxy running on the same machine, only one running somewhere else.
Use `--runtime docker` if your proxy runs locally.

The VM can't share a writable directory with the host, so `firecracker` only accepts read-only mounts (`--mount source:target:ro`).

### docker, podman, nerdctl

docker needs [buildx](https://github.com/docker/buildx#installing) to build the image.
On Ubuntu, install `docker.io` and `docker-buildx`; Docker's own packages from [its apt repository](https://docs.docker.com/engine/install/ubuntu/) include buildx already.
Then add yourself to the `docker` group (`sudo usermod -aG docker $USER`) and log in again.

podman builds on its own.
On macOS, start its VM first with `podman machine init` and `podman machine start`.

nerdctl needs [BuildKit](https://github.com/moby/buildkit): `buildctl` on your `PATH` and `buildkitd` running.
nerdctl's ["full" release](https://github.com/containerd/nerdctl/releases) bundles both.
Karotte checks for `buildctl` but not for `buildkitd`.

On a machine with `/dev/kvm`, the default runtime is `firecracker`, so pass `--runtime` to use one of these.

### modal

Install the extra and log in to Modal:

```sh
uv pip install 'karotte[modal]'
modal token new
```

Or set `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET`.

`karotte build --runtime modal` builds the image on Modal from the `Containerfile`; `--tag`, `--cache-from` and `--cache-to` are ignored.
`karotte run` builds the same way first, and with `--dev` reuses the last build.
A `--build-secret name=path` file is read as a dotenv file whose variables the build sees.

The `default` template's `KAROTTE_IMAGE_BUILDER` lines let Modal build the image: they skip file capabilities, which Modal's builder can't set, and add `nftables` for the student firewall.
An environment created before them needs them copied in (see [Updating](../environments/updating.md)); Karotte refuses to build it on Modal until then.

Each run gets its own sandbox under the Modal app `karotte`, sized like a VM (see [VM size](#vm-size)).
`--mount` paths are copied in when the run starts, and read-write ones and the transcript are copied back when it ends, as on `apple-container`.
The sandbox is stopped when the run ends, fails or is interrupted, and by Modal after 30 idle minutes if Karotte itself dies.
Cleanup only stops sandboxes your user started on the same machine.

A sandbox can't reach a model proxy on your machine, only one with a public address.
`--prepare-only`, `--keep-containers` and tasks on `passthrough` hardware, such as a GPU, aren't supported on `modal` yet.

### docker:gvisor

Install [gVisor](https://gvisor.dev/docs/user_guide/install/) and register `runsc` as a docker runtime in `/etc/docker/daemon.json` with these `runtimeArgs`:

```
-net-raw
--systrap-disable-syscall-patching
-overlay2=none
-file-access=shared
-network=sandbox
--net-disconnect-ok
```

Karotte checks this before each run.
If something is missing, it prints the exact `daemon.json` to write.
Then reload docker with `sudo systemctl reload docker`.

## VM size

A VM gets the CPUs, memory and disk that the `karotte.hardware_limits` plugin assigns to the task's hardware.
Without such a plugin, the sandbox gets 2 CPUs and 4 GiB of memory.

The student's memory limit is the sandbox's memory minus 1 GiB, which is kept for the harness.
The VM itself gets 1 GiB more than the sandbox's memory, for the guest kernel and page cache.
So without a plugin, the student has a 3 GiB limit inside a 5 GiB VM.
`apple-container` won't start a VM that's larger than your machine's RAM.

The student's disk budget is 80% of the host's free disk space, capped at the plugin's disk budget if it sets one.
With `karotte run -n N`, that budget is split evenly between the N runs.
Separate `karotte run` commands started at the same time don't know about each other, so each one counts all of the free space.
See [Student resources](student-resources.md) for how these limits are enforced.

## What each runtime gives the student

|                                                      | `apple-container`, `firecracker`                                            | `docker`/`podman`/`nerdctl` (runc)                                                  | `docker:gvisor`                                                                     |
| ---------------------------------------------------- | --------------------------------------------------------------------------- | ----------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------- |
| Memory limit                                         | Enforced by the kernel (cgroup)                                             | None by default. Watchdog if a plugin or `KAROTTE_SANDBOX_MEMORY_BYTES` sets a size | None by default. Watchdog if a plugin or `KAROTTE_SANDBOX_MEMORY_BYTES` sets a size |
| Process limit                                        | Enforced by the kernel (cgroup)                                             | Watchdog                                                                            | Watchdog                                                                            |
| Disk space and file count                            | Enforced by a quota filesystem                                              | Watchdog                                                                            | Watchdog                                                                            |
| Network                                              | Firewall, tested at startup. `firecracker` also filters traffic on the host | Firewall, tested at startup                                                         | Separate network namespace per session                                              |
| Processes that escape by forking a child and exiting | Separate PID namespace per session                                          | Found and killed by repeated sweeps                                                 | Separate PID namespace per session                                                  |

"Watchdog" means nothing stops the student from going over the limit, but Karotte notices and kills all of the student's processes.

`modal` gets the VM column: Modal's sandbox has its own kernel.
That kernel has no iptables owner match, so Karotte writes the student firewall as nftables rules there.
Modal's network around the sandbox isn't filtered the way `firecracker`'s is.

The `docker`/`podman`/`nerdctl` column is what you get in the usual case.
Karotte tries a writable cgroup first, so if the container has one, the kernel enforces the memory and process limits as it does in a VM.

On `firecracker`, the host-side filter blocks the whole guest, root included, from reaching link-local addresses, private address ranges and the host itself.
The model proxy is the only exception.
This holds even with `KAROTTE_STUDENT_NETWORK=internal`.

To see what your sandbox actually enforces, run `karotte check confinement` as root inside it.
It exits with 1 if the student gets less protection than it should:

```sh
karotte check confinement
karotte check confinement --json
```

It tests limits and firewall rules on an unused uid, mounts and unmounts a small file quota, and starts one student session under the default limits.
It undoes each of these afterwards, with one exception: it can leave the harness in the `karotte_harness` cgroup, which every run does at start anyway.
The report lists anything left behind.
`--hardware` picks which hardware's default memory limit to apply.

## Environment variables

Variables a launcher or task can set inside the sandbox:

| Variable                       | Effect                                                                                                                                                                                                                                                                                |
| ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `KAROTTE_SANDBOX`              | `runc`, `gvisor` or `vm`. Tells Karotte what kind of sandbox it's in, and takes precedence over detection. Without it, Karotte can only tell gVisor from runc, so a VM must set it. Karotte's own launchers set it.                                                                   |
| `KAROTTE_STUDENT_NETWORK`      | Unset or `strict`: the student can reach localhost, the sandbox's own addresses and the model proxy. `internal`: it can also reach link-local and private ranges, except on `firecracker`, where the host blocks them. Any other value counts as `strict`. Forwarded from your shell. |
| `KAROTTE_DISK_BUDGET_BYTES`    | Cap on the student's disk quota. Set by whoever knows the host's free space; VM launchers set it.                                                                                                                                                                                     |
| `KAROTTE_SANDBOX_MEMORY_BYTES` | The sandbox's total memory, when no plugin says. The student gets this minus 1 GiB, so a value of 1 GiB or less is ignored. VM launchers set it.                                                                                                                                      |
| `KAROTTE_VM_LAUNCHER`          | Set by Karotte's VM runtimes. `karotte check confinement` always fails if the VM has less memory than the sandbox's. With this variable set, it also fails if the VM lacks the extra 1 GiB for the guest kernel; without it, that only warns.                                         |
| `KAROTTE_FIREWALL_BACKEND`     | `nft` writes the student firewall as nftables rules instead of iptables ones. The `modal` runtime sets it, since Modal's kernel has no iptables owner match.                                                                                                                          |

Variables for the host:

| Variable                               | Effect                                                                                                                                     |
| -------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| `KAROTTE_FIRECRACKER_NETWORK`          | `pasta` (the default), or `none` for a VM with no network. Unset without pasta installed, the run stops instead of falling back to `none`. |
| `KAROTTE_FIRECRACKER_JAILER`           | `1` starts the VM through Firecracker's jailer. Needs root.                                                                                |
| `KAROTTE_FIRECRACKER_DNS`              | Comma-separated nameservers for the guest, instead of the host's.                                                                          |
| `KAROTTE_FIRECRACKER_WATCHDOG_SECONDS` | How long the guest may go without answering before Karotte kills its VM. Default 120; `0` turns it off.                                    |

`KAROTTE_STUDENT_NETWORK`, `KAROTTE_INFERENCE_SERVICE_TIER` and `LOGURU_LEVEL` are forwarded from your shell into the sandbox.
