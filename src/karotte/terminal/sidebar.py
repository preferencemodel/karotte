from collections.abc import Sequence
from typing import override

from textual.app import ComposeResult
from textual.widgets import ListItem, ListView, Static

from karotte.schemas.evaluation_run_config import EvaluationRunConfig


class Sidebar(ListView):
    """Sidebar showing all evaluation runs."""

    def __init__(
        self,
        configs: Sequence[EvaluationRunConfig],
        initial_index: int = 0,
        show_main_console: bool = True,
    ):
        super().__init__(initial_index=initial_index)
        self.configs: Sequence[EvaluationRunConfig] = configs
        self.show_main_console: bool = show_main_console
        self.run_items: list[NavItem] = []
        self.main_item: NavItem | None = None
        self.total_runs: int = len(configs)

    @override
    def compose(self) -> ComposeResult:
        # Add main console as first item if enabled
        if self.show_main_console:
            self.main_item = NavItem(
                icon="💻",
                label=f"Main (0/{len(self.configs)})",
                item_type="main",
            )
            yield self.main_item

        for i, config in enumerate(self.configs):
            item = NavItem(
                icon="",
                label=f"Run {config.run_id}",
                item_type="run",
                run_index=i,
                config=config,
            )
            self.run_items.append(item)
            # Items start hidden until runs are kicked off
            item.add_class("hidden")
            yield item

    def update_run_status(self, run_index: int, status: str) -> None:
        """Update the status of a specific run."""
        if 0 <= run_index < len(self.run_items):
            self.run_items[run_index].update_status(status)

    def update_run_color(self, run_index: int, run_status: str) -> None:
        """Update the color of a specific run based on its status."""
        if 0 <= run_index < len(self.run_items):
            self.run_items[run_index].update_color(run_status)

            # Update main console counter if run is completed
            if run_status in ("passed", "failed", "error") and self.main_item:
                completed = sum(
                    1
                    for item in self.run_items
                    if item.status in ("passed", "failed", "error", "completed")
                )
                # Update the main item's label with new counter
                try:
                    content = self.main_item.query_one(".nav-item-content", Static)
                    content.update(f"💻 Main ({completed}/{self.total_runs})")
                except Exception:
                    # Widget not mounted yet
                    pass


class NavItem(ListItem):
    """A unified navigation item for sidebar (Main console or Run)."""

    def __init__(
        self,
        icon: str,
        label: str,
        item_type: str = "main",  # "main" or "run"
        run_index: int | None = None,
        config: EvaluationRunConfig | None = None,
    ):
        super().__init__()
        self.icon: str = icon
        self.label: str = label
        self.item_type: str = item_type
        self.run_index: int | None = run_index
        self.config: EvaluationRunConfig | None = config
        self.status: str = "hidden" if item_type == "run" else "visible"
        self.loading_frame: int = 0

        # Apply initial status class
        if item_type == "run":
            self.add_class("status-running")
        elif item_type == "main":
            self.add_class("status-transparent")

    @override
    def compose(self) -> ComposeResult:
        yield Static(
            f"{self.icon} {self.label}",
            classes="nav-item-content",
        )

    def on_mount(self) -> None:
        """Update content after mounting to reflect any status changes."""
        self._update_content()

    def get_status_icon(self) -> str:
        """Get the status indicator for this item."""
        if self.item_type == "main":
            return "💻"

        # For runs
        if self.status == "hidden":
            return ""
        elif self.status in ("connecting", "connected", "disconnected"):
            # Animated loading indicator while connecting or running
            loading_frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
            return loading_frames[self.loading_frame % len(loading_frames)]
        elif self.status == "completed":
            return "✅"
        elif self.status == "failed":
            return "⛔"
        elif self.status == "error":
            return "⚠️ "
        return "?"

    def update_status(self, status: str) -> None:
        """Update the status and refresh the display.

        This is called from the websocket handler lifecycle. It must not
        override terminal statuses (completed/failed/error) that were set
        by update_color from the actual RunState.
        """
        if self.item_type != "run":
            return

        # Don't let websocket lifecycle override terminal run statuses
        if self.status in ("completed", "failed", "error"):
            return

        old_status = self.status
        self.status = status

        # Show/hide the item based on status
        if old_status == "hidden" and status != "hidden":
            self.remove_class("hidden")
        elif status == "hidden":
            self.add_class("hidden")

        # Update the content
        self._update_content()

    def update_color(self, run_status: str) -> None:
        """Update the background color based on run status."""
        if self.item_type != "run":
            return

        # Remove all status classes
        self.remove_class(
            "status-running", "status-passed", "status-failed", "status-error"
        )

        # Add appropriate class and update status for icon
        if run_status == "running":
            self.add_class("status-running")
        elif run_status == "passed":
            self.add_class("status-passed")
            self.status = "completed"
            self._update_content()
        elif run_status == "failed":
            self.add_class("status-failed")
            self.status = "failed"
            self._update_content()
        elif run_status == "error":
            self.add_class("status-error")
            self.status = "error"
            self._update_content()

        # Force refresh to make color change visible immediately
        self.refresh()

    def animate_loading(self) -> None:
        """Advance the loading animation frame."""
        if self.item_type == "run" and self.status in (
            "connecting",
            "connected",
            "disconnected",
        ):
            self.loading_frame += 1
            self._update_content()

    def _update_content(self) -> None:
        """Update the displayed content."""
        try:
            content = self.query_one(".nav-item-content", Static)
        except Exception:
            # Widget not mounted yet; will be updated when mounted
            return
        icon = self.get_status_icon()
        content.update(f"{icon} {self.label}")
