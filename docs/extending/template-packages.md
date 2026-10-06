# Template packages

Templates can live in their own Python package.
Karotte finds them through the `karotte.templates` entry point and treats them like its own templates.
They show up in `karotte templates list`, they stack with `default`, and `karotte update` keeps them up to date.
See [Templates](../environments/templates.md) for how templates stack.

## Layout

Put the template directories under one directory in your package.
Each template directory contains a `template.toml` and the files the template renders:

```text
my_package/
  __init__.py
  templates/
    rust/
      template.toml
      Containerfile
      src/environment/tasks/rust_task/__init__.py
```

Register the directory under the `karotte.templates` entry point:

```toml
[project.entry-points."karotte.templates"]
my_templates = "my_package:TEMPLATES_DIR"
```

`TEMPLATES_DIR` is the path to that directory:

```python
from pathlib import Path

TEMPLATES_DIR = Path(__file__).parent / "templates"
```

Every subdirectory of `TEMPLATES_DIR` is a template.
The subdirectory's name is the template id.
It can contain lowercase letters, digits and hyphens, but it can't start or end with a hyphen.

## `template.toml`

`template.toml` holds the fields of [`EnvironmentTemplate`](https://github.com/preferencemodel/karotte/blob/main/src/karotte/schemas/environment_template.py), except `id`, which comes from the directory name.
Only `description` is required:

```toml
description = "Adds a Rust toolchain."
requires = ["default"]
```

## Jinja

Karotte renders every file in a template with Jinja, not only `.jinja` files.
If a file contains text with `{{` or `{%` in it, wrap that text in `{% raw %}` … `{% endraw %}`.

Templates have access to two variables:

| Variable    | Value                                              |
| ----------- | -------------------------------------------------- |
| `env_name`  | The name of the environment's directory.           |
| `templates` | The ids of all templates being rendered, in order. |

Karotte drops a `.jinja` suffix from the output path.
Use the suffix for files that aren't valid on their own, such as a `pyproject.toml` that starts with `{% extends %}`.
Karotte never copies `template.toml` itself.

To change a file from another template, put a file at the same path that extends it and overrides its blocks:

```jinja
{% extends "default/Containerfile" %}
{% block extra_build_steps -%}
RUN dnf install -y cargo
{% endblock %}
```

The `default` template has these blocks:

| File                                                         | Blocks                                                                               |
| ------------------------------------------------------------ | ------------------------------------------------------------------------------------ |
| `Containerfile`                                              | `builder_stages`, `extra_env_vars`, `extra_system_dependencies`, `extra_build_steps` |
| `CLAUDE.md`                                                  | `claude_md`                                                                          |
| `pyproject.toml.jinja`, `venvs/student/pyproject.toml.jinja` | `head`, `uv_extra`, `indexes`, `tail`                                                |

### Partials

Partials let several templates add to the same file without overriding each other's blocks.
Karotte never copies the files under a template's `partials/` directory into the environment.
Instead, `default` includes them from every template being rendered that has them:

| Partial                        | Included in                                                                                         |
| ------------------------------ | --------------------------------------------------------------------------------------------------- |
| `partials/deps.toml.jinja`     | `[project] dependencies` of the environment's `pyproject.toml`                                      |
| `partials/dev_deps.toml.jinja` | the `dev` extra of the environment's `pyproject.toml`                                               |
| `partials/uv_extra.toml.jinja` | the `[tool.uv]` table of the environment's and the venvs' `pyproject.toml`                          |
| `partials/indexes.toml.jinja`  | the `[[tool.uv.index]]` entries of the environment's and the venvs' `pyproject.toml`, ahead of PyPI |
| `partials/Containerfile.jinja` | the end of the `Containerfile`, before `karotte check` runs                                         |

For example, this `partials/deps.toml.jinja` adds a dependency:

```jinja
    "numpy",
```

## Recipes

The `default` template's `justfile` imports `internal.just` if that file exists:

```just
import? 'internal.just'
```

Use it for recipes that belong to your own setup rather than to Karotte, such as deploying to your own infrastructure.
If a template ships an `internal.just`, every environment created from that template gets its recipes.
You can also create `internal.just` by hand in a single environment.
`karotte update` doesn't touch files that no template ships.

A recipe in `internal.just` can't reuse the name of a recipe in the `justfile`.
`just` refuses to run if a name is defined twice.

!!! warning

    If you create `internal.just` by hand and later add a template that ships one, `karotte update` replaces your file with the template's version.

## Installing

Install the package next to Karotte when you create the environment:

```sh
uvx --with my-package karotte create-env my_env --template rust
```

Karotte records the package as `name==version` in the environment's [manifest](../environments/templates.md#the-manifest) (`extra_deps`).
That way, `karotte update` installs it again, at its newest release, to re-render the templates.
If the environment didn't use your package when it was created, pass the package once with `karotte update --with my-package`.
See [Updating environments](../environments/updating.md).

`karotte update` resolves template packages against the environment's own package indexes.
Those are the non-explicit `[[tool.uv.index]]` entries in its `pyproject.toml`.
If your package isn't on PyPI, it can add its index through `partials/indexes.toml.jinja`.

## Conflicts and failures

- If a package provides a template with the same id as one of Karotte's own, Karotte uses the package's template instead and logs a warning.
- If two packages provide the same template id, that's an error.
- If a package's entry point fails to load, or one of its template directories has a missing or invalid `template.toml`, Karotte skips all of that package's templates with a warning.
