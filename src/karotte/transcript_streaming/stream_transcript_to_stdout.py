import json
import sys
import warnings
from collections.abc import AsyncIterator
from typing import Any

from karotte.schemas.transcript import (
    AnswersSubmittedEvent,
    ErrorEvent,
    Event,
    MessageAddedEvent,
    MessageChunkEvent,
    MessageChunkResetEvent,
    ScoringEvent,
    StepCompletedEvent,
    StepStartedEvent,
    TaskCompletedEvent,
    TaskPreHookCompletedEvent,
    TaskStartedEvent,
    TokenUsageEvent,
    ToolCallCompletedEvent,
    ToolCallStartedEvent,
)
from karotte.truncation import truncate_middle
from rich.console import Console
from rich.markup import escape

_MAX_DISPLAY_CHARS = 100_000

# rich's highlighter is quadratic on long unbroken tokens; markup still works.
print = Console(highlight=False).print


async def stream_transcript_to_stdout(event_stream: AsyncIterator[Event]):
    # https://github.com/fastapi/sqlmodel/discussions/1369
    warnings.filterwarnings(
        "ignore",
        message=".*Accessing the 'model_fields' attribute on the instance is deprecated.*",
    )
    warnings.filterwarnings(
        "ignore",
        message=".*The `dict` method is deprecated; use `model_dump` instead.*",
    )

    new_message = True
    line_open = False
    n_steps = -1
    step = 0
    steps_passed = 0
    final_score = None

    def end_streamed_line() -> None:
        nonlocal line_open
        if line_open:
            sys.stdout.write("\n")
            sys.stdout.flush()
            line_open = False

    async for event in event_stream:
        match event:
            case TaskStartedEvent():
                n_steps = event.n_steps
                print(f"[blue]{'━' * 80}[/blue]")
                print(f"[blue]Task: {escape(event.task_id)}[/blue]")
                print(f"[blue]{'━' * 80}[/blue]")

            case TaskPreHookCompletedEvent():
                print("Executed task pre hook")
                print("Pre hook metadata: ")
                for key, value in event.metadata.items():
                    _print_kv(key, value)

            case StepStartedEvent():
                step = event.step
                print(f"\n\n Starting step {event.step + 1}")

            case MessageChunkEvent():
                if event.delta.content:
                    if new_message:
                        _ = print("\n🗣️  Student:")
                        new_message = False
                    sys.stdout.write(event.delta.content)
                    sys.stdout.flush()
                    line_open = not event.delta.content.endswith("\n")

            case MessageChunkResetEvent():
                end_streamed_line()
                new_message = True

            case MessageAddedEvent():
                end_streamed_line()
                new_message = True
                if event.message.role == "user" and event.message.content:
                    _ = print(
                        f"\n👤 User: {escape(_truncate(event.message.content))}\n"
                    )

            case ToolCallStartedEvent():
                match event.tool_call.function.name:
                    case "bash":
                        print("\n\n🔧 Calling Bash:")
                        try:
                            arguments = json.loads(event.tool_call.function.arguments)
                            for line in str(arguments["command"]).splitlines():
                                print(escape(line))
                        except (json.JSONDecodeError, KeyError):
                            print(
                                f"Invalid arguments: {escape(repr(event.tool_call.function.arguments))}"
                            )

                        print()
                    case _:
                        print(
                            f"\n\n🔧 Calling tool: {escape(event.tool_call.function.name or '')} {escape(event.tool_call.function.arguments)}"
                        )

            case ToolCallCompletedEvent():
                print("✅ Tool call completed:")
                if event.result.structuredContent:
                    for key, value in event.result.structuredContent.items():
                        print(f"{escape(str(key))}:\n{escape(_truncate(value))}")

            case AnswersSubmittedEvent():
                print("\n\n✅ Answers submitted:")
                for key, value in event.answers.items():
                    _print_kv(key, value)

            case ScoringEvent():
                scoring = event.scoring
                final_score = scoring.score
                steps_passed += scoring.continue_task
                print("\n\n✅ Scoring completed:")
                print("[bold]Score:[/bold]", scoring.score)

                if step + 1 != n_steps:
                    print(
                        f"\n[bold]Continue task? -> {'Yes' if scoring.continue_task else 'No'}[/bold]"
                    )

                if scoring.metadata:
                    print("\n[bold]Metadata:[/bold]")
                    for key, value in scoring.metadata.items():
                        _print_kv(key, value)

                if event.resource_metrics and event.resource_metrics.samples:
                    print(
                        f"\n[bold]Resource Metrics:[/bold]\n{escape(event.resource_metrics.model_dump_json())}"
                    )

                print("\n\n")

            case ErrorEvent():
                print("\n\n[bold red]💥 Error occurred:[/bold red]")
                print(f"[red]Exception Type: {escape(event.exception_type)}[/red]")
                print(f"[red]Message: {escape(_truncate(event.message))}[/red]")
                if event.traceback:
                    print("\n[red]Full Traceback:[/red]")
                    print(f"[red]{escape(_truncate(event.traceback))}[/red]\n")

            case StepCompletedEvent():
                pass
            case TaskCompletedEvent():
                total = f"/{n_steps}" if n_steps >= 0 else ""
                score = "N/A" if final_score is None else final_score
                print(
                    f"\n[bold]🏁 Result: {event.status}, {steps_passed}{total} steps passed, final score {score}[/bold]"
                )
            case TokenUsageEvent():
                pass
            case _:
                print(
                    f"[bold red]Unknown event type: {type(event).__name__}[/bold red]"
                )


def _print_kv(key: Any, value: Any) -> None:
    """Print a key-value pair with Rich markup escaping."""
    print(f"{escape(str(key))}: {escape(_truncate(value))}")


def _truncate(value: Any) -> str:
    return truncate_middle(str(value), _MAX_DISPLAY_CHARS)
