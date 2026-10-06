---
hide:
    - toc
    - path
---

# Karotte

Karotte is an open-source framework for building robust RL environments, made by [Preference Model](https://preferencemodel.com).

You write tasks in Python.
Karotte builds them into a sandbox, lets an agent work inside it through tools like `bash`, scores what it did, and records every message, tool call and score in a transcript.
The agent runs as an unprivileged user with resource limits, a firewall and a disk quota, so a task can hand it a real shell without trusting it.

```sh
uv tool install karotte
karotte create-env my_env
```

<div class="grid cards" markdown>

- :lucide-rocket:{ .lg .middle } **Get started**

    ***

    Install Karotte, create an environment and run the example task.

    [:lucide-arrow-right: Quick start](getting-started/quick-start.md)

- :lucide-list-checks:{ .lg .middle } **Write tasks**

    ***

    Steps, judges, tools, and the data and dependencies a task needs.

    [:lucide-arrow-right: Tasks and steps](tasks/tasks-and-steps.md)

- :lucide-shield:{ .lg .middle } **Run in a sandbox**

    ***

    VMs and containers, what each one enforces, and the limits on the agent.

    [:lucide-arrow-right: Runtimes](running/runtimes.md)

- :lucide-puzzle:{ .lg .middle } **Extend Karotte**

    ***

    Your own templates, plugins, and a backend that collects runs.

    [:lucide-arrow-right: Plugins](extending/plugins.md)

</div>
