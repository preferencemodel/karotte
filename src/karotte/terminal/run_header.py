from collections.abc import Sequence
from typing import override

from textual.app import ComposeResult
from textual.containers import Container
from textual.reactive import reactive
from textual.widgets import Static

from karotte.schemas.evaluation_run_config import EvaluationRunConfig
from karotte.schemas.run_state import RunState


class RunHeader(Container):
    """Fixed header showing metadata for the selected run."""

    run_index: reactive[int | None] = reactive(None)

    def __init__(
        self,
        configs: Sequence[EvaluationRunConfig],
        *,
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
    ):
        super().__init__(name=name, id=id, classes=classes)
        self.configs: Sequence[EvaluationRunConfig] = configs

    @override
    def compose(self) -> ComposeResult:
        yield Static("", id="run-header-content", markup=False)

    def on_mount(self) -> None:
        """Update display when widget is mounted."""
        if self.run_index is not None:
            self.watch_run_index(self.run_index)

    def watch_run_index(self, run_index: int | None) -> None:
        """Update header when selected run changes."""
        content_widget = self.query_one("#run-header-content", Static)

        if run_index is None:
            content_widget.update("No run selected")
            return

        if 0 <= run_index < len(self.configs):
            config = self.configs[run_index]
            content = (
                f"ID: {config.run_id}  |  "
                f"Task: {config.task_id}  |  "
                f"Model: {config.model}"
            )
            content_widget.update(content)

    def update_from_run_state(
        self,
        run_state: RunState,
        view_mode: str = "json",  # pyright: ignore[reportUnusedParameter]
    ) -> None:
        """Update header with information from RunState."""
        content_widget = self.query_one("#run-header-content", Static)

        parts = [
            f"ID: {run_state.run_id}",
            f"Task: {run_state.task_id}",
        ]

        if run_state.status == "running":
            if run_state.n_steps >= 1:
                parts.append(f"Step: {run_state.current_step + 1}/{run_state.n_steps}")
            parts.append(
                f"Score: {run_state.score if run_state.score is not None else 'N/A'}"
            )
        elif run_state.status == "passed":
            parts.append("Status: ✓ Passed")
            parts.append(
                f"Score: {run_state.score if run_state.score is not None else 'N/A'}"
            )
        elif run_state.status == "failed":
            parts.append("Status: ✗ Failed")
            parts.append(
                f"Score: {run_state.score if run_state.score is not None else 'N/A'}"
            )
        elif run_state.status == "error":
            parts.append("Status: ⚠ Error")
            parts.append(
                f"Score: {run_state.score if run_state.score is not None else 'N/A'}"
            )

        content_widget.update("  |  ".join(parts))
