# Scratch envs for the template tests. Not under /tmp: Apple `container`
# builds lose directory contents from a context under a symlinked path.
test_root := env_var("HOME") + "/.cache/karotte/test-templates"

_default:
  @just --list

# Static analysis
lint:
  uv run --frozen --extra dev ruff format --check
  uv run --frozen --extra dev ruff check
  uv run --frozen --extra dev basedpyright
  uv run --frozen --extra dev scripts/check_supply_chain_config.py

# Auto-format and apply automatic fixes for linting
fix:
  uv run --frozen --extra dev ruff format
  uv run --frozen --extra dev ruff check --fix

# Preview the docs at http://localhost:8000
docs *args:
  uv run --frozen --group docs zensical serve {{ args }}

# Build the docs into site/, failing on broken links
docs-build:
  uv run --frozen --group docs zensical build --clean --strict

# Run tests for karotte
test *args:
  uv run --frozen --extra test pytest {{ args }}

# Run the `requires_root` tests as root. Serial: they share uids.
test-root:
  sudo -E env HOME=/root PATH="$PATH" \
    "$(uv run --frozen --extra test python -c 'import sys; print(sys.executable)')" \
    -m pytest -m requires_root --numprocesses=0 -p no:cacheprovider \
    $(grep -l requires_root tests/test_*.py)

# Run tests for templates
test-template template *run_args:
  rm -rf {{ test_root }}/test_{{ template }}
  mkdir -p {{ test_root }}

  uv run --frozen karotte create-env {{ test_root }}/test_{{ template }} --template {{ template }} --vendor-karotte

  cd {{ test_root }}/test_{{ template }} && uv run setup_data.py

  cd {{ test_root }}/test_{{ template }} && uv run --extra dev just lint
  cd {{ test_root }}/test_{{ template }} && uv run --extra dev just test
  cd {{ test_root }}/test_{{ template }} && uv run karotte create-run-config --model claude-haiku-4-5-20251001
  cd {{ test_root }}/test_{{ template }} && uv run karotte run --config run_config.json --no-ui --no-proxy {{ run_args }}
  cd {{ test_root }}/test_{{ template }} && uv run python {{ justfile_directory() }}/tests/check_transcript.py out/transcript.json

# `test-template` for language-toolchains
test-toolchains-template container_cmd="docker" *run_args:
  rm -rf {{ test_root }}/test_language-toolchains
  mkdir -p {{ test_root }}

  uv run --frozen karotte create-env {{ test_root }}/test_language-toolchains --template language-toolchains --vendor-karotte

  cd {{ test_root }}/test_language-toolchains && uv run setup_data.py

  cd {{ test_root }}/test_language-toolchains && uv run --extra dev just lint
  cd {{ test_root }}/test_language-toolchains && uv run --extra dev just test
  cd {{ test_root }}/test_language-toolchains && uv run --extra dev just enable-all-toolchains
  cd {{ test_root }}/test_language-toolchains && uv run karotte create-run-config --model claude-haiku-4-5-20251001
  cd {{ test_root }}/test_language-toolchains && uv run karotte run --config run_config.json --no-ui --no-proxy {{ run_args }}
  cd {{ test_root }}/test_language-toolchains && uv run python {{ justfile_directory() }}/tests/check_transcript.py out/transcript.json
  cd {{ test_root }}/test_language-toolchains && uv run --extra dev just check-toolchains karotte '' '{{ container_cmd }}'
