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

# Run tests for karotte
test:
  uv run --frozen --extra test pytest

# Run the `requires_root` tests as root. Serial: they share uids.
test-root:
  sudo -E env HOME=/root PATH="$PATH" \
    "$(uv run --frozen --extra test python -c 'import sys; print(sys.executable)')" \
    -m pytest -m requires_root --numprocesses=0 -p no:cacheprovider tests

# Run tests for templates
test-template template *run_args:
  rm -rf /tmp/test_{{ template }}

  uv run --frozen karotte create-env /tmp/test_{{ template }} --template {{ template }} --vendor-karotte

  cd /tmp/test_{{ template }} && uv run setup_data.py

  cd /tmp/test_{{ template }} && uv run --extra dev just lint
  cd /tmp/test_{{ template }} && uv run --extra dev just test
  cd /tmp/test_{{ template }} && uv run karotte create-run-config --model claude-haiku-4-5-20251001
  cd /tmp/test_{{ template }} && uv run karotte run --config run_config.json --no-ui --no-proxy {{ run_args }}
  cd /tmp/test_{{ template }} && uv run python {{ justfile_directory() }}/tests/check_transcript.py out/transcript.json

# `test-template` for language-toolchains
test-toolchains-template container_cmd="docker" *run_args:
  rm -rf /tmp/test_language-toolchains

  uv run --frozen karotte create-env /tmp/test_language-toolchains --template language-toolchains --vendor-karotte

  cd /tmp/test_language-toolchains && uv run setup_data.py

  cd /tmp/test_language-toolchains && uv run --extra dev just lint
  cd /tmp/test_language-toolchains && uv run --extra dev just test
  cd /tmp/test_language-toolchains && uv run --extra dev just enable-all-toolchains
  cd /tmp/test_language-toolchains && uv run karotte create-run-config --model claude-haiku-4-5-20251001
  cd /tmp/test_language-toolchains && uv run karotte run --config run_config.json --no-ui --no-proxy {{ run_args }}
  cd /tmp/test_language-toolchains && uv run python {{ justfile_directory() }}/tests/check_transcript.py out/transcript.json
  cd /tmp/test_language-toolchains && uv run --extra dev just check-toolchains karotte '' '{{ container_cmd }}'
