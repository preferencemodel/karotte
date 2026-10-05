# Concepts

## Environments and the karotte library

An _environment_ is the isolated world a model (we call it the _student_) works in: the tasks, the tools it can use, the data and dependencies it needs, and the judges that score it.
You define it as a Python project, created with `karotte create-env`, and karotte builds it into a container image that runs in a sandbox.
In the Python project it is the `environment` package, which depends on the `karotte` library.

karotte provides everything an environment needs:

- the harness that runs a task, talks to the student and scores the result
- tools for the student, such as `bash`, `view_lines_in_file` and `replace_in_file`
- judges, such as `ExecutableJudge` and `RubricJudge`
- transcripts, streamed to the terminal UI or a backend
- the `karotte` CLI, which builds images, runs tasks and shows transcripts

The environment adds everything else on top:

- its tasks, in `src/environment/tasks/`
- a `Containerfile` that builds it into a container image
- data and dependencies for the student and for scoring

Since karotte is a dependency, you get its fixes and features by updating it; see [Updating](../environments/updating.md).
The files `create-env` generates come from [templates](../environments/templates.md).

## Terminology

| Term       | Meaning                                                                                                                                                                                                                                                                         |
| ---------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Student    | The model under test. Its commands run as an unprivileged `student` user with resource limits, a firewall and a disk quota. See [Student resources](../running/student-resources.md).                                                                                           |
| Task       | A verifiable unit of work that the student must complete. See [Tasks and steps](../tasks/tasks-and-steps.md).                                                                                                                                                                   |
| Step       | One part of a task. A step consists of instructions for the student and a judge that scores the result.                                                                                                                                                                         |
| Judge      | Decides how well the student did in a step. It returns a score and whether the task should continue. See [Scoring](../tasks/scoring.md).                                                                                                                                        |
| Agent      | What drives the student through the task. The `builtin` agent calls the model itself through [litellm](https://docs.litellm.ai/); others get the model's messages from a backend or run a CLI coding agent baked into the image. See [Agents](../running/run-config.md#agents). |
| Image      | The environment built into a container image by `karotte build/run`.                                                                                                                                                                                                            |
| Sandbox    | One copy of the image executing one run: a VM or a container, depending on the runtime.                                                                                                                                                                                         |
| Runtime    | What runs the image: a VM (`apple-container`, `firecracker`) or a container engine (`docker`, `podman`, `docker:gvisor`). See [Runtimes](../running/runtimes.md).                                                                                                    |
| Run config | A JSON file that names the task, the model, its API key and options such as hints and limits. See [Run config](../running/run-config.md).                                                                                                                                       |
| Transcript | The record of a run: every message, tool call and score. See [Artifacts and transcripts](../tasks/artifacts-and-transcripts.md).                                                                                                                                                |
