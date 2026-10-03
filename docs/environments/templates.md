# Templates

A template is a directory of files.
`karotte create-env` renders those files with [Jinja](https://jinja.palletsprojects.com/) to make a new environment.
karotte ships two templates:

| Template              | What it gives you                                                                                                           |
| --------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| `default`             | A CPU environment with an example task. Every other template builds on it.                                                  |
| `language-toolchains` | Experimental. Gives the student exactly one language toolchain per task. See [Language toolchains](language-toolchains.md). |

`karotte templates list` shows every installed template, including those from [template packages](../extending/template-packages.md).
It also shows which templates each one requires.
Add `--json` to get the same list as JSON.

```text
$ karotte templates list
Available templates:
  default — A CPU environment with an example task.
  language-toolchains — Securely provide exactly one language toolchain to the student. Used by environments that vary tasks by programming language.
    Requires: default
```

## Creating an environment

```sh
karotte create-env my_env
```

If you don't pass `--template`, `create-env` uses `default`.
The target directory must not exist yet.
After rendering, `create-env` runs `uv lock`.
Then it runs the environment's `post_create.py`, if there is one.
The `default` template's `post_create.py` locks every venv under `venvs/`.
If any of these steps fails, `create-env` removes the new directory again.

## What `default` contains

| Path                                                                                           | Purpose                                                                                                                                                           |
| ---------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `src/environment/tasks/`                                                                       | Your tasks. `example_task` shows submissions, hints and hooks. `_template` is what `just create-task` copies. See [Tasks and steps](../tasks/tasks-and-steps.md). |
| `src/environment/tools/`, `src/environment/judges/`                                            | Your own tools and judges.                                                                                                                                        |
| `Containerfile`                                                                                | Builds the image.                                                                                                                                                 |
| `setup_data.py`, `student_data/`, `shared_data/`, `root_data/`, `intermediate_data/`, `venvs/` | Data and extra venvs for the image. See [Data and dependencies](../tasks/data-and-dependencies.md).                                                               |
| `justfile`                                                                                     | `lint`, `fix`, `fmt`, `test`, `lock-venvs`, `create-task`, `vendor-karotte` and more. Run `uv run just --list` to see them all.                                   |
| `tests/`                                                                                       | Tests for the environment.                                                                                                                                        |
| `CLAUDE.md`                                                                                    | Instructions for coding agents that work on the environment.                                                                                                      |
| `.manifest.json`                                                                               | Records how the environment was made. See [The manifest](#the-manifest).                                                                                          |

The templates are under the MIT No Attribution license (`MIT-0`).
Environments you create from them don't need a license notice.

## Stacking templates

To stack templates, pass `--template` more than once:

```sh
karotte create-env my_env --template default --template language-toolchains
```

karotte renders `default` first and then renders `language-toolchains` on top of it.
karotte always renders a template's `requires` before the template itself.
So `karotte create-env my_env --template language-toolchains` gives you the same result.

A later template can replace a file completely by shipping its own file at the same path.
It can also extend a file from an earlier template and override only some of its Jinja blocks:

```jinja
{% extends "default/CLAUDE.md" %}
{% block claude_md -%}
New content
{% endblock %}
```

The `extends` path is the template id followed by the file's path inside that template.
Include the `.jinja` suffix if the file has one, as in `default/pyproject.toml.jinja`.
[Writing a template package](../extending/template-packages.md#jinja) lists the blocks that `default` offers.

## The manifest

Every environment has a `.manifest.json` file that records how it was made:

| Field                        | Meaning                                                               |
| ---------------------------- | --------------------------------------------------------------------- |
| `karotte_version`            | The karotte release that rendered the environment.                    |
| `templates`                  | The templates in the order they were applied, including dependencies. |
| `agents`                     | CLI agents baked into the image (`create-env --agent`).               |
| `do_not_recreate_if_deleted` | Paths that `karotte update` won't bring back after you delete them.   |
| `extra_deps`                 | The `name==version` of each package that provided a template.         |

`karotte update` reads the manifest to re-render the environment and merge in template changes.
When it's done, it writes the manifest back.
See [Updating environments](updating.md).

`do_not_recreate_if_deleted` starts out as the combined lists of all the environment's templates.
For `default`, that list is `CLAUDE.md` and `src/environment/tasks/example_task`.
So if you delete the example task, it stays deleted.
You can add paths to the list or remove them.
An entry for a directory covers everything under it.
