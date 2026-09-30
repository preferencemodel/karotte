- Use `from loguru import logger` for logging.
- Fix formatting and lint issues and then run linting: `just fix && just lint`

- You can test things by creating a test environment with `karotte create-env` and building its container image with `karotte build`.

## How to add a model

A new Claude release in an existing family (a point release like `claude-sonnet-5-1`, a dated snapshot) needs only step 1 and step 5: `_CLAUDE_ADAPTIVE_PREFIXES` in `model_spec.py` matches it by prefix, and every bare `claude-*` id is sent to litellm as `anthropic/<id>`. A new Claude family gets its prefix added there (and to `_NO_SAMPLING_PARAMS_PREFIXES` if it rejects temperature). Everything else follows all five steps.

1. Add the id to `CATALOG_MODEL_IDS` in `src/karotte/model_catalog.py`.
2. Describe it in `src/karotte/model_spec.py`: max output tokens, the provider's lowest/highest `reasoning_effort` names, and how to get reasoning back (`_reasoning_request`).
   Check the provider's docs and litellm's model map; don't guess. If litellm rejects a param for a model it doesn't know yet, force it through with `_EXTRA_ALLOWED_OPENAI_PARAMS`.
   The picker's `model_display_name` is derived from the id by `_model_display_name`; check it in the `model_catalog.json` golden diff and fix the rule if it reads wrong. A new provider gets one `_PROVIDER_DISPLAY_NAMES` entry.
3. Verify against the real API before trusting the docs. Call it with curl and with `litellm.completion` (directly or through your `--proxy`), streaming, with a tool call, and check that reasoning comes back as `reasoning_content`.
4. Add parametrized cases in `tests/test_model_spec.py` (effort range, max tokens, litellm model name) and, when litellm needs special handling, a test in `tests/test_model_goldens.py`.
5. Regenerate every golden that dumps model specs and review the diff; only the new model should change:
   `UPDATE_GOLDEN=1 pytest tests/test_model_goldens.py tests/test_model_catalog.py tests/test_characterization.py`
