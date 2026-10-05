{% raw -%}
# Recipes for the language-toolchains template; the environment justfile pulls
# these in with `import?`.

# Build a hello-world with every enabled language toolchain, inside the built
# image. One container per language: installing a toolchain is not reversible.
# Pass a language to check just that one, and a runtime like "podman"
# where docker is not the engine. `jobs` languages run at once; each one's
# output is held back and printed whole when it finishes.
check-toolchains image="karotte" language="" runtime="docker" jobs="4":
  #!/usr/bin/env bash
  set -euo pipefail
  languages="{{ language }}"
  if [ -z "$languages" ]; then
    languages=$(uv run --frozen python scripts/check_toolchains.py --list)
  fi
  if [ -z "$languages" ]; then
    echo "No languages enabled in src/environment/toolchain_config.py; nothing to check."
    exit 0
  fi
  # A build gets a mount namespace of its own and refuses to run without one,
  # so a container that cannot make one fails every check before it reaches a
  # compiler. CAP_SYS_ADMIN is what `unshare` and `mount` want; the default
  # AppArmor profile denies mount(2) whatever the capabilities say.
  check() {
    local lang="$1" log status=0
    log=$(mktemp)
    {{ runtime }} run --rm \
      --volume "$PWD/scripts:/scripts:ro" \
      --cap-add=SYS_ADMIN \
      --security-opt seccomp=unconfined \
      --security-opt apparmor=unconfined \
      --entrypoint /root/.venv/bin/python \
      "{{ image }}" /scripts/check_toolchains.py "$lang" >"$log" 2>&1 || status=$?
    cat "$log"
    rm -f "$log"
    return $status
  }
  export -f check
  printf '%s\n' $languages | xargs -P "{{ jobs }}" -I{} bash -c 'check "$1"' _ {}

# Rewrite toolchain_config.py to enable every language in the table. Meant for
# CI, which builds the everything-image; environments pick their own subset.
enable-all-toolchains:
  #!/usr/bin/env bash
  set -euo pipefail
  uv run --frozen python - <<'EOF'
  import re
  from pathlib import Path

  from environment.toolchains import Language

  path = Path("src/environment/toolchain_config.py")
  names = ", ".join(f'"{language.value}"' for language in sorted(Language))
  text, count = re.subn(
      r"^ENABLED_LANGUAGES.*$",
      "ENABLED_LANGUAGES: frozenset[str] = frozenset({" + names + "})",
      path.read_text(),
      flags=re.MULTILINE,
  )
  assert count == 1, f"expected one ENABLED_LANGUAGES line, found {count}"
  path.write_text(text)
  print(f"Enabled all {len(Language)} languages in {path}")
  EOF
{%- endraw %}
