"""Tests for sidebar NavItem status behavior.

The sidebar icon and color must be driven by RunState (via update_color),
not by the websocket connection lifecycle (via update_status).
"""

from karotte.terminal.sidebar import NavItem

LOADING_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]


def _make_run_item() -> NavItem:
    return NavItem(icon="", label="Run test", item_type="run", run_index=0)


def _make_main_item() -> NavItem:
    return NavItem(icon="💻", label="Main", item_type="main")


class TestGetStatusIcon:
    """get_status_icon returns the correct icon for each status."""

    def test_main_item_always_returns_laptop(self):
        item = _make_main_item()
        assert item.get_status_icon() == "💻"

    def test_hidden_returns_empty(self):
        item = _make_run_item()
        item.status = "hidden"
        assert item.get_status_icon() == ""

    def test_connecting_shows_loading(self):
        item = _make_run_item()
        item.status = "connecting"
        assert item.get_status_icon() in LOADING_FRAMES

    def test_connected_shows_loading(self):
        item = _make_run_item()
        item.status = "connected"
        assert item.get_status_icon() in LOADING_FRAMES

    def test_disconnected_shows_loading(self):
        """While disconnected (scoring may still be running), show loading."""
        item = _make_run_item()
        item.status = "disconnected"
        assert item.get_status_icon() in LOADING_FRAMES

    def test_disconnected_does_not_show_green_check(self):
        """Websocket disconnecting must NOT show the green check."""
        item = _make_run_item()
        item.status = "disconnected"
        assert item.get_status_icon() != "✅"

    def test_completed_shows_green_check(self):
        item = _make_run_item()
        item.status = "completed"
        assert item.get_status_icon() == "✅"

    def test_failed_shows_stop(self):
        item = _make_run_item()
        item.status = "failed"
        assert item.get_status_icon() == "⛔"

    def test_error_shows_warning(self):
        item = _make_run_item()
        item.status = "error"
        assert "⚠️" in item.get_status_icon()

    def test_unknown_status_shows_question_mark(self):
        item = _make_run_item()
        item.status = "something_unexpected"
        assert item.get_status_icon() == "?"

    def test_loading_frame_advances(self):
        """Different loading_frame values produce different icons."""
        item = _make_run_item()
        item.status = "connected"
        item.loading_frame = 0
        icon_a = item.get_status_icon()
        item.loading_frame = 1
        icon_b = item.get_status_icon()
        assert icon_a != icon_b
        assert icon_a in LOADING_FRAMES
        assert icon_b in LOADING_FRAMES


class TestUpdateStatusDoesNotOverrideTerminalState:
    """update_status (from websocket) must not override terminal statuses set by update_color."""

    def test_disconnected_does_not_override_completed(self):
        item = _make_run_item()
        item.status = "completed"
        item.update_status("disconnected")
        assert item.status == "completed"

    def test_disconnected_does_not_override_failed(self):
        item = _make_run_item()
        item.status = "failed"
        item.update_status("disconnected")
        assert item.status == "failed"

    def test_disconnected_does_not_override_error(self):
        item = _make_run_item()
        item.status = "error"
        item.update_status("disconnected")
        assert item.status == "error"

    def test_connected_does_not_override_completed(self):
        item = _make_run_item()
        item.status = "completed"
        item.update_status("connected")
        assert item.status == "completed"

    def test_connecting_does_not_override_failed(self):
        item = _make_run_item()
        item.status = "failed"
        item.update_status("connecting")
        assert item.status == "failed"

    def test_disconnected_updates_from_connected(self):
        """Transitioning from connected to disconnected is fine."""
        item = _make_run_item()
        item.status = "connected"
        item.update_status("disconnected")
        assert item.status == "disconnected"

    def test_connected_updates_from_connecting(self):
        item = _make_run_item()
        item.status = "connecting"
        item.update_status("connected")
        assert item.status == "connected"

    def test_noop_for_main_item(self):
        item = _make_main_item()
        item.status = "visible"
        item.update_status("disconnected")
        assert item.status == "visible"

    def test_hidden_to_connecting_unhides(self):
        item = _make_run_item()
        item.status = "hidden"
        item.add_class("hidden")
        item.update_status("connecting")
        assert item.status == "connecting"
        assert "hidden" not in item.classes

    def test_transition_to_hidden_hides(self):
        item = _make_run_item()
        item.status = "connected"
        item.update_status("hidden")
        assert item.status == "hidden"
        assert "hidden" in item.classes


class TestUpdateColor:
    """update_color (from RunState) sets the terminal status and CSS class."""

    def test_passed_sets_completed_status(self):
        item = _make_run_item()
        item.update_color("passed")
        assert item.status == "completed"

    def test_failed_sets_failed_status(self):
        item = _make_run_item()
        item.update_color("failed")
        assert item.status == "failed"

    def test_error_sets_error_status(self):
        item = _make_run_item()
        item.update_color("error")
        assert item.status == "error"

    def test_running_does_not_set_terminal_status(self):
        item = _make_run_item()
        item.status = "connected"
        item.update_color("running")
        assert item.status == "connected"

    def test_passed_adds_status_passed_class(self):
        item = _make_run_item()
        item.update_color("passed")
        assert "status-passed" in item.classes

    def test_failed_adds_status_failed_class(self):
        item = _make_run_item()
        item.update_color("failed")
        assert "status-failed" in item.classes

    def test_error_adds_status_error_class(self):
        item = _make_run_item()
        item.update_color("error")
        assert "status-error" in item.classes

    def test_running_adds_status_running_class(self):
        item = _make_run_item()
        item.update_color("running")
        assert "status-running" in item.classes

    def test_removes_previous_status_classes(self):
        item = _make_run_item()
        item.update_color("running")
        assert "status-running" in item.classes
        item.update_color("passed")
        assert "status-running" not in item.classes
        assert "status-passed" in item.classes

    def test_noop_for_main_item(self):
        item = _make_main_item()
        item.update_color("passed")
        assert item.status == "visible"


class TestAnimateLoading:
    """animate_loading advances the frame for active loading statuses."""

    def test_advances_frame_when_connecting(self):
        item = _make_run_item()
        item.status = "connecting"
        item.loading_frame = 0
        item.animate_loading()
        assert item.loading_frame == 1

    def test_advances_frame_when_connected(self):
        item = _make_run_item()
        item.status = "connected"
        item.loading_frame = 0
        item.animate_loading()
        assert item.loading_frame == 1

    def test_advances_frame_when_disconnected(self):
        item = _make_run_item()
        item.status = "disconnected"
        item.loading_frame = 0
        item.animate_loading()
        assert item.loading_frame == 1

    def test_does_not_advance_when_completed(self):
        item = _make_run_item()
        item.status = "completed"
        item.loading_frame = 0
        item.animate_loading()
        assert item.loading_frame == 0

    def test_does_not_advance_when_failed(self):
        item = _make_run_item()
        item.status = "failed"
        item.loading_frame = 0
        item.animate_loading()
        assert item.loading_frame == 0

    def test_does_not_advance_for_main_item(self):
        item = _make_main_item()
        item.loading_frame = 0
        item.animate_loading()
        assert item.loading_frame == 0


class TestEndToEndStatusFlow:
    """Test realistic sequences of update_status/update_color calls."""

    def test_websocket_disconnects_before_scoring_completes(self):
        """The bug scenario: websocket closes before TaskCompletedEvent is processed."""
        item = _make_run_item()
        item.update_status("connecting")
        item.update_status("connected")
        # Websocket closes while scoring is still running
        item.update_status("disconnected")
        assert item.get_status_icon() in LOADING_FRAMES
        assert item.get_status_icon() != "✅"
        # Scoring finishes, RunState sends passed
        item.update_color("passed")
        assert item.status == "completed"
        assert item.get_status_icon() == "✅"

    def test_scoring_completes_before_websocket_disconnects(self):
        """Normal fast-scoring case: RunState resolves before websocket closes."""
        item = _make_run_item()
        item.update_status("connecting")
        item.update_status("connected")
        # RunState processes TaskCompletedEvent
        item.update_color("passed")
        assert item.status == "completed"
        # Websocket closes after — should not change anything
        item.update_status("disconnected")
        assert item.status == "completed"
        assert item.get_status_icon() == "✅"

    def test_run_fails_then_websocket_disconnects(self):
        """Run fails, then websocket disconnects — should stay failed."""
        item = _make_run_item()
        item.update_status("connecting")
        item.update_status("connected")
        item.update_color("failed")
        assert item.status == "failed"
        item.update_status("disconnected")
        assert item.status == "failed"
        assert item.get_status_icon() == "⛔"

    def test_websocket_disconnects_then_run_fails(self):
        """Websocket closes, then scoring resolves as failed."""
        item = _make_run_item()
        item.update_status("connecting")
        item.update_status("connected")
        item.update_status("disconnected")
        assert item.get_status_icon() in LOADING_FRAMES
        item.update_color("failed")
        assert item.status == "failed"
        assert item.get_status_icon() == "⛔"

    def test_websocket_disconnects_without_task_completed_shows_error(self):
        """If websocket disconnects and no TaskCompletedEvent ever arrives, show error.

        The app layer detects this (disconnected + queue drained + no terminal
        RunState) and calls update_color("error").
        """
        item = _make_run_item()
        item.update_status("connecting")
        item.update_status("connected")
        item.update_status("disconnected")
        assert item.get_status_icon() in LOADING_FRAMES
        # App detects no terminal status after queue drain
        item.update_color("error")
        assert item.status == "error"
        assert "⚠️" in item.get_status_icon()
