---
name: harbor-to-karotte
description: >-
  Port Harbor-format benchmark tasks (task.toml, instruction.md,
  environment/Dockerfile, tests/, solution/), such as Terminal-Bench, into
  karotte environments, one environment per task. Use when asked to convert,
  port or import Harbor tasks or a Harbor dataset into karotte.
---

# Harbor → karotte

Turn Harbor tasks into karotte environments that behave like the original: the
student sees the same task, with the same tools and limits, and is scored by
the same rules.

[terminal-bench-4-karotte-port](https://github.com/preferencemodel/terminal-bench-4-karotte-port)
is a worked example: 65 Terminal-Bench 4.0 tasks ported this way. Every task
there has its own Containerfile, data setup and scoring, so porting is
per-task work, not a format conversion. Read the port of a task similar to
yours before you start.

## 1. Get the dataset and pin versions

- Work from a local copy of the tasks: a folder, or a git clone at a fixed
  commit or tag.
- The Harbor hub serves single files at
  `https://hub.harborframework.com/api/tasks/<org>/<task>/<rev>/files/<path>`
  but has no listing endpoint. Use it as a download source for files you
  already know about; check that each URL returns the file you expect.
- Pin one karotte version (`uvx karotte@latest --version`) and use it for
  every task.

## 2. Triage

```sh
python skills/harbor-to-karotte/scripts/inventory.py <dataset-dir> --json inventory.json
```

This prints one row per task (hardware, timeouts, flags) with a first-guess
verdict: `portable`, `needs-adaptation`, or `skip-or-defer`. The flags show
where the work is:

| Flag | Meaning |
|---|---|
| `gpus=N` | karotte's VM runtimes can't pass a GPU through yet. |
| `custom-base`, `multi-stage` | The toolchain must be rebuilt on the karotte image (Amazon Linux 2023). |
| `build-downloads` | The Dockerfile downloads things; pin and hash-check them. |
| `multi-container`, `service-check` | Sidecar services, or tests that talk to a running service. |
| `root-in-solution`, `installs-in-solution` | The karotte student is unprivileged and has no network. |
| `network-in-tests` | The tests reach the network. |
| `verifier-image`, `conftest`, `extra-dirs` | Notes, not blockers: read `tests/Dockerfile`, `conftest.py`, or folders outside the Harbor format. |

The script only reads text, so read each task before agreeing with its
verdict. Write a short triage report: counts, the work each flag implies, and
the tasks you'd skip, with reasons. Skip a task whose point doesn't survive the
port (GPU-only, root administration) rather than gutting it.

For more than a few tasks, agree on an order and a budget first. Port one task
end to end, review it, then do the rest.

## 3. Port each task

Start from `karotte create-env <out>/<task>` and read the whole upstream task:
`task.toml`, `instruction.md`, the Dockerfile and its files, every test, and
the solution. Then:

- **Instruction**: `instruction.md` becomes the step's instructions. Harbor
  works in `/app`; the karotte student works in `/workdir`, so rewrite paths.
  If the whole of `/app` is the deliverable, use `/workdir/app`.
- **Image**: add the task's system packages and toolchains to the
  `Containerfile`, pinned and hash-checked. Put Python packages the student
  needs in the student venv. The student has no network, so preinstall
  anything the solution installs.
- **Data**: in `setup_data.py`, stage the files the student starts with, and
  put hidden grading data and the upstream tests in `root_data/`, each pinned
  by sha256. Keep the solution out of the image.
- **Limits**: `[environment]` becomes `required_hardware`;
  `[agent].timeout_sec` goes into the instructions; `[verifier].timeout_sec`
  becomes the judge's timeout.
- **Scoring**: one named check per upstream test, with upstream's reward rule
  (usually 1.0 only if every test passes). Run student code as the student,
  never in the grader. `submission_paths` must cover everything the tests
  read, because `collect_submission` deletes the rest. Every failure must
  score 0, not crash.

Keep upstream's behavior unless it can't work in karotte, or it is a bug that
changes what's measured. Write every deviation down.

## 4. Verify

For each task:

- `uv run setup_data.py` passes its sha256 pins;
- the task's unit tests pass, and none of them was skipped (`pytest -rs`);
- `just lint` passes;
- `karotte build` succeeds (`--runtime docker` on macOS, or `--runtime modal`);
- in the sandbox, with the fake model, the reference solution scores 1.0 and
  an empty run scores 0.0.

Report each check as passed, failed, or not run with the reason.

## 5. Write it up

A `README.md` for the ported dataset: how to run a task, hardware notes, the
skipped tasks with reasons, and a credit table that links each task to its
upstream and lists its authors from `task.toml`. Keep the upstream license and
say the tasks are modified versions of the originals.
