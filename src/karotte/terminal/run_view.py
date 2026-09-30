from collections.abc import Sequence
from typing import override

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.reactive import reactive

from karotte.schemas.evaluation_run_config import EvaluationRunConfig
from karotte.terminal.console_view import ConsoleView
from karotte.terminal.run_header import RunHeader
from karotte.terminal.transcript_view import TranscriptView


class RunView(Vertical):
    """Right side panel showing details and transcript for selected run."""

    run_index: reactive[int | None] = reactive(None)
    show_main_console: reactive[bool] = reactive(False)

    def __init__(
        self,
        configs: Sequence[EvaluationRunConfig],
        *,
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
        disabled: bool = False,
    ):
        super().__init__(name=name, id=id, classes=classes, disabled=disabled)
        self.configs: Sequence[EvaluationRunConfig] = configs

    @override
    def compose(self) -> ComposeResult:
        yield RunHeader(self.configs, id="run-header")
        yield ConsoleView(id="main-console-view", classes="hidden")
        yield TranscriptView(id="transcript-view")

    def on_mount(self) -> None:
        """Update display when widget is mounted."""
        if self.run_index is not None:
            self.watch_run_index(self.run_index)

    def watch_run_index(self, run_index: int | None) -> None:
        """Update detail view when selected run changes."""
        header = self.query_one("#run-header", RunHeader)
        transcript = self.query_one("#transcript-view", TranscriptView)

        header.run_index = run_index
        transcript.clear_transcript()

    def watch_show_main_console(self, show_main: bool) -> None:
        """Toggle between main console and transcript view."""
        main_console = self.query_one("#main-console-view", ConsoleView)
        transcript = self.query_one("#transcript-view", TranscriptView)
        header = self.query_one("#run-header", RunHeader)

        if show_main:
            # Show main console, hide transcript and header
            main_console.remove_class("hidden")
            transcript.add_class("hidden")
            header.add_class("hidden")
        else:
            # Show transcript, hide main console
            main_console.add_class("hidden")
            transcript.remove_class("hidden")
            header.remove_class("hidden")
