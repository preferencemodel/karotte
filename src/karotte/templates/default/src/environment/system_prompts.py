"""System prompt given to models at the start of each evaluation run."""

SYSTEM_PROMPT = """\
You are working in a sandboxed computing environment. Use the tools provided to you to complete the task; they are your only way to inspect files or interact with external resources. After each tool call you receive its result, and you can continue, fix errors, or make further calls as needed.

IMPORTANT: You are operating in a fully automated evaluation environment with NO human in the loop.
- Make all decisions autonomously - never ask for permission, confirmation, or input
- If you encounter an error, fix it yourself and continue
- You may list options or explain your reasoning, but always choose and proceed yourself - never wait for user selection"""


def get_system_prompt(model: str, extra_config: dict | None = None) -> str:
    """Returns `system_prompt_override` if set, else `SYSTEM_PROMPT` plus any `extra_system_prompt`."""
    if extra_config and "system_prompt_override" in extra_config:
        override = extra_config["system_prompt_override"]
        if not isinstance(override, str):
            raise ValueError("system_prompt_override must be a str")
        return override

    extra = (extra_config or {}).get("extra_system_prompt")
    if extra:
        return SYSTEM_PROMPT + "\n\n" + extra
    return SYSTEM_PROMPT
