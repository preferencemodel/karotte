import json

from karotte import EvaluationRunConfig
from karotte.schemas import ChatCompletionMessageToolCall, Function, Message

from environment.paths import STUDENT_DATA_DIR


def get_messages(config: EvaluationRunConfig) -> list[Message]:  # pyright: ignore[reportUnusedParameter]
    # A step keeps running until a message has no tool call, so end each step
    # with one. Give it text too. Turns without text or tool calls count as stalled
    # and get nudged instead of ending the step.
    return [
        # Step 1
        Message(
            role="assistant",
            content="path: /workdir/.venv/bin/python",
        ),
        # Step 2
        Message(
            role="assistant",
            content="And the Python version.",
            tool_calls=[
                ChatCompletionMessageToolCall(
                    id="tool_call_1",
                    type="function",
                    function=Function(
                        name="bash",
                        arguments=json.dumps(
                            {
                                "command": f"echo '3.12.11' > {STUDENT_DATA_DIR}/python_version.txt"
                            }
                        ),
                    ),
                )
            ],
        ),
        Message(role="assistant", content="Done."),
    ]
