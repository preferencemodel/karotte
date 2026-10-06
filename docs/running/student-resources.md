# Student resources

Karotte limits how much memory, how many processes and how many files the student can use.
This stops a runaway or hostile student from starving the harness and crashing the run, for example when your grader runs the student's code.

## The default limits

Every task gets three limits, applied when the task starts, right before `pre_hook`:

- **Memory**: the sandbox's memory minus 1 GiB, which is kept for the harness.
  Karotte takes the sandbox's memory from the first of these that gives an answer:
  the `karotte.hardware_limits` [plugin](../extending/plugins.md) for the task's `required_hardware`, `KAROTTE_SANDBOX_MEMORY_BYTES` (which the VM runtimes set), the cgroup limit above the student, or the machine's RAM.
  If cgroups don't work and neither the plugin nor the variable gives a size, there's no memory limit.
  In a VM without a plugin, the student gets 3 GiB; see [Runtimes](runtimes.md#vm-size).
- **Processes**: 2048.
- **Files**: 80% of the free disk space when the task starts, and at most 1,000,000 files.
  The byte budget is capped at `KAROTTE_DISK_BUDGET_BYTES` if the launcher set it, otherwise at the plugin's disk budget.
  It covers everywhere the student can write to real disk: the workdir, and any temp directory (`/tmp`, `/var/tmp`, `/dev/shm`) that isn't stored in RAM.

## Changing the limits

You can change the limits at any time with `limit_resources`, usually from `pre_hook`.
Limits you don't pass keep their current value.
Passing `None` removes a limit.

```python
from karotte import FileLimit, get_resource_limits, limit_resources

# Raise the memory cap; the process cap stays as it was
limit_resources(memory_bytes=12 * 1024**3)

# Remove the process cap
limit_resources(process_count=None)

# Cap the student's files under the workdir at 5 GiB and 100k files
limit_resources(file=FileLimit(path="/workdir", bytes=5 * 1024**3, count=100_000))

# The limits currently in effect
get_resource_limits()
```

`FileLimit.path` also accepts a tuple of paths.
The limits apply to the student by default; pass `uid=` to limit a different uid.
Outside a Karotte container, `limit_resources` does nothing.

`get_resource_limits()` returns a `ResourceLimits` with `memory_bytes`, `process_count` and `file`.
It reads them from whatever enforces each limit, and uses `None` where there's no limit.

## How limits are enforced

`limit_resources` returns a `Contract` for each limit it set, under the keys `memory`, `processes` and `files`:

| Contract              | Meaning                                                                                                                            |
| --------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| `prevented`           | The kernel enforces the limit. An allocation or fork that would go over it fails.                                                  |
| `detected_and_reaped` | Nothing stops the student from going over the limit, but a watchdog notices and kills every process the student owns with SIGKILL. |
| `not_supported`       | Nothing enforces the limit in this sandbox.                                                                                        |

Karotte uses the strongest method the sandbox supports.
By default, each runtime gives:

| Limit     | `apple-container`, `firecracker` | `docker`/`podman` (runc) | `docker:gvisor`       |
| --------- | -------------------------------- | ------------------------ | --------------------- |
| Memory    | `prevented`                      | `detected_and_reaped`    | `detected_and_reaped` |
| Processes | `prevented`                      | `detected_and_reaped`    | `detected_and_reaped` |
| Files     | `prevented`                      | `detected_and_reaped`    | `detected_and_reaped` |

[Runtimes](runtimes.md#what-each-runtime-gives-the-student) covers the network and process isolation each runtime adds.
`karotte check confinement` shows what your sandbox actually enforces.

Why each runtime ends up where it does:

- A VM has its own kernel with writable cgroups, so the kernel enforces its memory and process caps.
- runc mounts the cgroups read-only, so it falls back to the watchdog.
- gVisor accepts cgroup settings but doesn't enforce them, so it always uses the watchdog.
- For the file limit, Karotte creates a filesystem exactly as large as the budget, in both bytes and number of files, and mounts it over the directories the student can write to.
  A write that doesn't fit fails with "no space left on device".
  Only files written during the run count; files that were already in the image stay visible and don't use up the budget.
  This needs root and loop devices, which a VM has.
- If a task changes the file limit later, the watchdog enforces the new numbers, because a mounted filesystem can't shrink.
  The filesystem stays mounted, so its size is still a hard upper limit.
- If there's no usable cgroup and no student uid to watch, the limit is `not_supported`.

### What the watchdog guarantees

The watchdog only notices a limit was crossed after it happens.
It can't make an allocation or a fork fail.

- It checks memory and processes every 0.1 seconds, so the student can briefly go over a cap before being caught.
  It checks files once per second, because it has to look at every file the student owns.
- When it catches a violation, it kills every process the student owns, not just the one that went over.
- It keeps running after it kills the student's processes.
  A violation can outlast the processes that caused it: files in RAM-backed temp directories and SysV shared-memory segments survive a SIGKILL.
  As long as the violation lasts, every new process the student starts is killed too, until the files are deleted or the limit is removed.
- It first measures memory with RSS, which counts shared memory once for every process that shares it.
  Before killing anything, it checks again with an exact measurement (PSS), so a job that forks many workers isn't killed for memory it only holds once.
- It also kills the student's processes when it can't get a reliable reading, which happens when the RAM-backed temp directories hold more than 10,000 entries.
- If a task removes the process cap, the memory check stops counting after 10,000 processes instead of killing the student.
  Memory spread across more processes than that is partly invisible to the memory cap.
- If a task removes the file-count cap, the file check stops counting after 1,000,000 entries instead of killing the student.
  Data spread across more files than that is partly invisible to the byte cap.

## What counts as memory

The student's memory includes more than what its processes hold:

- the resident memory of every process the student owns
- files in the RAM-backed temp directories (`/tmp`, `/var/tmp`, `/dev/shm` when they're tmpfs)
- SysV shared-memory segments the student created

The last two are RAM that stays in use after the process that created it is gone, so killing every student process doesn't free them.
Under a cgroup, the kernel does its own accounting, and it also charges tmpfs pages to the group that wrote them.

When each student launch gets its own IPC namespace, the watchdog can't see that launch's SysV segments from outside.
So a detached segment isn't counted while the launch is still running.
It can't outlive the launch, though: the namespace frees it when the launch's last process exits.

## What isn't limited

Karotte doesn't limit the student's CPU use.
A VM runtime gives the whole VM a fixed number of CPUs; see [Runtimes](runtimes.md#vm-size).

## Killing student processes

Before grading, kill everything the student owns with `kill_processes`.
The `default` template does this in `pre_scoring_hook`:

```python
from karotte import kill_processes

from environment import STUDENT_UID

kill_processes(STUDENT_UID)
```

If the student has a real cgroup, `kill_processes` first kills the whole group in a single atomic kernel operation, which a fork bomb can't escape by spawning new processes.
It also kills the student's processes in PID namespaces below Karotte's, which takes down everything in a namespace at once.
Then a helper running as the student's uid kills everything that uid owns with `kill(-1, SIGKILL)`, and Karotte checks `/proc` for anything that survived.
It repeats this until a pass finds nothing left.

If it can't confirm that the student's processes are gone, it raises an error instead of returning.
If processes are still alive after 30 seconds, it raises `UnreapableCohortError`.
That's a `StudentMisbehaviorError`, so in `pre_scoring_hook` it scores the step 0 instead of failing the run with an error; see [Scoring](../tasks/scoring.md).
If it can't run the kill as the student's uid at all, because it isn't running as root, it raises a plain `RuntimeError`.
Grading must never run while a student process is still alive.

## Cleaning up what the student left behind

Once the student's processes are gone, `delete_files(uid)` removes what that uid still has: every file it owns anywhere on the filesystem, and the SysV shared-memory segments it created.
It returns how many things it removed.
Karotte doesn't call it for you.
Call it from a hook, after saving any files you still need for grading.

```python
from pathlib import Path

from karotte import delete_files

from environment import STUDENT_UID

delete_files(STUDENT_UID, extend_exclude=(Path("/workdir/data/answer.txt"),))
```

`include` limits the search to some directories.
`exclude` replaces the default paths to keep, `/root` and `/var/karotte_quota`, and `extend_exclude` adds to them.
[`reclaim.py`](https://github.com/preferencemodel/karotte/blob/main/src/karotte/reclaim.py) has all the arguments and their defaults.

An excluded path is kept along with everything under it and the directories leading to it.
Other files next to it are still deleted.
Read-only and virtual filesystems are skipped, and so are the paths in `KAROTTE_RECLAIM_EXCLUDE` (separated by `:`).
Outside a Karotte container, `delete_files` refuses to search the whole filesystem, because the uid might belong to a real user of the machine.

`delete_files` raises `ReclaimError` if it can't guarantee that it deleted everything the uid owned, or if it ran past `timeout`.
The timeout exists because a student can leave behind more files than Karotte can go through in a reasonable time.
`ReclaimError` is also a `StudentMisbehaviorError`, so in `pre_scoring_hook` either case scores the step 0.

A common use of `extend_exclude` is freeing up disk space before copying a large submission; see [Scoring](../tasks/scoring.md).
