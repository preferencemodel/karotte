# Updating environments

`karotte update` brings an environment up to the latest karotte release and its templates.
It merges the template changes into your files, so your own edits survive.

Run it from the environment's directory:

```sh
uvx karotte update
```

`uvx` runs the newest karotte.
You can also pass the environment's directory: `uvx karotte update path/to/my_env`.
Commit your work first, so that `git diff` shows what the update changed.
`karotte update` needs `git` on your `PATH`.

## How it works

1. karotte reads `.manifest.json` (see [The manifest](templates.md#the-manifest)) to find the karotte version and templates the environment was made with.
2. It looks up the newest karotte release on the environment's package indexes.
   There is nothing to do if all of these hold: that release is the version in the manifest, no template package has a newer release, and you aren't adding a template.
3. It renders the environment twice into a temporary directory, both times with the manifest's templates.
   The first render uses the old karotte and is called the _baseline_.
   The second uses the new karotte and is called the _target_.
4. It applies the difference between the two renders to your environment, one file at a time.
5. It writes the new version, templates and template packages to the manifest.
6. It re-locks the environment and any venv whose `pyproject.toml` changed.

In step 4, each file is handled like this:

| Template change      | Your file      | Result                                                                    |
| -------------------- | -------------- | ------------------------------------------------------------------------- |
| Changed text file    | Present        | 3-way merge with `git merge-file`. Overlapping changes become a conflict. |
| Changed binary file  | Present        | Replaced with the template's version.                                     |
| File in both renders | Deleted by you | Recreated, unless it's under `do_not_recreate_if_deleted`.                |
| New file             | Absent         | Added, unless it's under `do_not_recreate_if_deleted`.                    |
| Removed file         | Present        | Deleted.                                                                  |

!!! warning

    If the template no longer ships a file, the update deletes it from your environment, even if you changed it.
    Check `git status` after an update.

karotte never merges `.git`, `.venv`, `__pycache__`, `.ruff_cache`, `uv.lock` or `.manifest.json`.

The manifest's `do_not_recreate_if_deleted` list gets merged too.
Paths that a new template version adds to its list are added to yours.
Paths that it drops from its list are removed from yours.
Your own edits to the list stay.

## Conflicts

At the end, `karotte update` lists the files that have conflicts.
Each conflict has the usual markers, with your version first and the template's version second.

```text
<<<<<<< ...
your lines
=======
the template's lines
>>>>>>> ...
```

Edit each file until it looks the way you want, and remove the markers.
Sometimes `git` can't merge a file at all.
In that case karotte keeps your version, reports the file as a conflict, and logs the reason.
You then need to apply the template's change by hand.

## Re-locking

If `pyproject.toml` merged without conflicts, karotte runs:

```sh
uv lock --upgrade-package karotte
```

It adds an extra `--upgrade-package` for each template package.
A plain `uv lock` would keep the karotte version that's already in the lock, and then the lock and the manifest would disagree.

karotte also runs `uv lock` in each `venvs/*/` directory whose `pyproject.toml` the merge added or changed.

If `pyproject.toml` has conflicts, it isn't valid TOML.
In that case karotte skips the re-lock and prints the command for you to run.
Once you've resolved the conflicts, run that command yourself.
Also run `uv lock` in each `venvs/*/` directory that had conflicts.

## Adding a template

To add a template to an environment that was made without it, pass `--add-template`:

```sh
uvx karotte update --add-template language-toolchains
```

karotte renders the baseline with the manifest's templates, and the target with the new template added.
As a result, the new template's files arrive as additions, and its changes to existing files arrive as merges.
karotte resolves the new template's `requires` too, and records the result in the manifest.
You can pass `--add-template` more than once.
If you name a template that's already in the manifest, nothing happens.
If you name the same template twice, you get an error.
The update also brings the environment up to the latest karotte.

## Templates from other packages

karotte renders templates from a [template package](../extending/template-packages.md) with that package installed next to karotte.
The manifest records the package in `extra_deps`, and `karotte update` installs the package's newest release for the target render.
If a recorded package isn't installed next to the karotte you're running, `karotte update` restarts itself with that package.

Sometimes the manifest doesn't record a template's package yet, for example when you add the template with `--add-template`.
In that case, pass the package with `--with`:

```sh
uvx karotte update --with my-package --add-template rust
```

## When `update` refuses to run

- One of `UV_INDEX`, `UV_DEFAULT_INDEX`, `UV_INDEX_URL` or `UV_EXTRA_INDEX_URL` is set.
  uv prefers these over the indexes in the environment's `pyproject.toml`, so the update would lock against the wrong index.
  Unset them.
- The environment has no `.manifest.json`.
- karotte can't find the newest release, or it can't render one of the versions.
  The error includes uv's output.
  If a template isn't found, pass the package that provides it with `--with`.

karotte looks up releases on the environment's own non-explicit `[[tool.uv.index]]` entries.
It uses the same 7-day age limit as the environment's `exclude-newer` setting.
karotte itself and template packages are exempt from that limit.
