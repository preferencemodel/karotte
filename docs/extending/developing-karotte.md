# Developing Karotte

This page covers working on Karotte itself, and testing a changed Karotte inside an environment.
Before you change Karotte for the sake of one environment, check whether a [plugin](plugins.md), a [template package](template-packages.md) or the environment's own code can do what you need.
If none of them can, open an issue or a pull request on [GitHub](https://github.com/preferencemodel/karotte).

## Working on Karotte

```sh
git clone https://github.com/preferencemodel/karotte.git
cd karotte
uv sync --extra dev
```

The repo's `justfile` has recipes for the common tasks:

| Command                                         | What it does                                                                                                       |
| ----------------------------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| `just lint`                                     | Runs `ruff format --check`, `ruff check`, `basedpyright` and the supply-chain config check.                        |
| `just fix`                                      | Formats the code and applies automatic lint fixes.                                                                 |
| `just test [args]`                              | Runs the tests with pytest. Extra arguments are passed on to pytest.                                               |
| `just test-root`                                | Runs the tests marked `requires_root` as root, one at a time.                                                      |
| `just test-template <template> [run args]`      | Creates an environment from a template, then lints it, tests it and runs its task end to end.                      |
| `just test-toolchains-template [container cmd]` | Does the same for `language-toolchains`, with every language enabled, and runs `just check-toolchains` at the end. |
| `just docs`                                     | Serves this documentation at `http://localhost:8000`.                                                              |
| `just docs-build`                               | Builds the documentation into `site/` and fails on broken links.                                                   |

`just test-template default` creates the environment under `~/.cache/karotte/test-templates/` with `--vendor-karotte`, so the environment runs your checkout.
It calls a real model, so you need `ANTHROPIC_API_KEY`.
Any arguments after the template name are passed on to `karotte run`, for example `just test-template default --runtime docker`.

## Releases

Every merge to `main` is released to PyPI once CI passes.
The major and minor version come from `pyproject.toml`.
The patch version is the number of commits on `main`.

## Using your Karotte in an environment

To try a Karotte change in an environment, you vendor Karotte.
That means copying its source into the environment's `.karotte/` directory and pointing the environment's `pyproject.toml` at it.
The `default` template's `Containerfile` copies `.karotte/` into the image, so the image runs your copy.

!!! warning

    Vendoring is an escape hatch for testing a change or shipping an urgent fix.
    Once the change is merged, switch back to the released Karotte.

### `just vendor-karotte`

In an environment based on the `default` template:

```sh
just vendor-karotte                   # clone main from GitHub
just vendor-karotte ~/code/karotte    # copy a local checkout
just vendor-karotte my-feature-branch # clone a branch, tag or commit
```

The recipe does four things:

1. Replaces `.karotte/` with a copy of the local checkout, or with a clone of the git ref with its `.git/` removed.
2. Adds `karotte = { path = ".karotte" }` under `[tool.uv.sources]` in `pyproject.toml` and comments out any earlier `karotte` source.
3. Edits `.gitignore` so you can commit `.karotte/`.
4. Deletes `.venv` and runs `uv lock`.

Without the `[tool.uv.sources]` entry, uv installs Karotte from the index and ignores `.karotte/`.

Then edit `.karotte/src/karotte/` and rebuild the image.
You can rebuild with `uv run karotte build`, or with any `karotte run` that doesn't pass `--dev`.
A `--dev` run won't pick up the change, because `--dev` mounts only `src/environment/`.

After you make more changes in your local checkout, run `just vendor-karotte ~/code/karotte` again.
It replaces `.karotte/`, deletes `.venv` and locks again.

To go back to the released Karotte:

```sh
just unvendor-karotte
```

This empties `.karotte/`, restores `pyproject.toml` and `.gitignore`, deletes `.venv` and runs `uv lock`.

### `karotte create-env --vendor-karotte`

`create-env --vendor-karotte` vendors whichever Karotte runs the command into the new environment.
To get an environment that runs your code, run it from your checkout:

```sh
cd ~/code/karotte
uv run karotte create-env ~/envs/my_env --vendor-karotte
```

It copies the source to `.karotte/` and sets the `[tool.uv.sources]` entry, just like `just vendor-karotte` does.
Unlike the recipe, it leaves `.karotte/` ignored in `.gitignore`.

### Templates aren't vendored

`create-env --vendor-karotte` and `just vendor-karotte <local path>` don't copy Karotte's `templates/` directory.
If you run `karotte create-env` or `karotte templates list` from such a copy, it stops and asks you to install Karotte as a tool (`uv tool install karotte`).
Run those commands with a Karotte installed as a tool, or from your checkout.
