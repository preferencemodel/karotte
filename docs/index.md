---
hide:
  - toc
  - path
---

# karotte

karotte runs LLM agents on tasks and scores the results.

You write tasks in Python.
karotte builds them into a sandbox, lets the model work inside it through tools like `bash`, scores what it did, and records every message, tool call and score in a transcript.
The model runs as an unprivileged user with resource limits, a firewall and a disk quota, so a task can hand it a real shell without trusting it.

```sh
uv tool install karotte
karotte create-env my_env
```

<div class="grid cards" markdown>

-   :lucide-rocket:{ .lg .middle } **Get started**

    ---

    Install karotte, create an environment and run the example task.

    [:lucide-arrow-right: Quick start](getting-started/quick-start.md)

-   :lucide-list-checks:{ .lg .middle } **Write tasks**

    ---

    Steps, judges, tools, and the data and dependencies a task needs.

    [:lucide-arrow-right: Tasks and steps](tasks/tasks-and-steps.md)

-   :lucide-shield:{ .lg .middle } **Run in a sandbox**

    ---

    VMs and containers, what each one enforces, and the limits on the model.

    [:lucide-arrow-right: Runtimes](running/runtimes.md)

-   :lucide-puzzle:{ .lg .middle } **Extend karotte**

    ---

    Your own templates, plugins, and a backend that collects runs.

    [:lucide-arrow-right: Plugins](extending/plugins.md)

</div>
