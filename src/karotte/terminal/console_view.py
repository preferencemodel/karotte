from typing import override

from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.css.query import NoMatches
from textual.widgets import Static


class ConsoleView(VerticalScroll):
    """Scrollable view showing main console output (stdout/stderr)."""

    def __init__(
        self,
        *,
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
        disabled: bool = False,
    ):
        super().__init__(name=name, id=id, classes=classes, disabled=disabled)
        self.auto_scroll: bool = True
        self._pending_lines: list[str] = []

    @override
    def compose(self) -> ComposeResult:
        yield Static("", id="console-content", markup=False)

    def on_mount(self) -> None:
        """Flush any pending lines that arrived before mount."""
        if self._pending_lines:
            for line in self._pending_lines:
                self._append_line_to_widget(line)
            self._pending_lines.clear()

    def _append_line_to_widget(self, line: str) -> None:
        """Append a line directly to the widget (must be mounted)."""
        content = self.query_one("#console-content", Static)
        current = str(content.render())

        if current == "":
            content.update(line)
        else:
            content.update(f"{current}\n{line}")

        if self.auto_scroll:
            self.scroll_end(animate=False)

    def append_line(self, line: str) -> None:
        """Append a line to the console output."""
        try:
            content = self.query_one("#console-content", Static)
        except NoMatches:
            # Widget not mounted yet; buffer the line for later
            self._pending_lines.append(line)
            return
        # Get the current text content
        current = str(content.render())

        if current == "":
            content.update(line)
        else:
            content.update(f"{current}\n{line}")

        if self.auto_scroll:
            self.scroll_end(animate=False)
