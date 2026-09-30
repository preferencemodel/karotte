import shlex
import subprocess
import traceback
from collections import deque
from collections.abc import Callable, Sequence
from functools import partial
from io import StringIO
from pathlib import Path
from typing import override

import anyio
from loguru import logger
from pydantic import TypeAdapter
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.css.query import NoMatches
from textual.widgets import Footer, Header

from karotte import Runtime
from karotte.build import get_container_build_command
from karotte.load_tasks import load_task
from karotte.run_helpers import (
    clean_up_old_containers,
    get_container_run_command,
    run_non_containerized,
)
from karotte.runtime import get_engine
from karotte.schemas.evaluation_run_config import EvaluationRunConfig
from karotte.schemas.run_state import RunState
from karotte.schemas.transcript import (
    Event,
    MessageAddedEvent,
    MessageChunkEvent,
    MessageChunkResetEvent,
    TaskCompletedEvent,
    TaskStartedEvent,
    ToolCallCompletedEvent,
    Transcript,
)
from karotte.schemas.websocket_config import WebSocketConfig
from karotte.terminal.console_view import ConsoleView
from karotte.terminal.discover_transcripts import discover_transcripts
from karotte.terminal.run_header import RunHeader
from karotte.terminal.run_view import RunView
from karotte.terminal.sidebar import Sidebar
from karotte.terminal.transcript_view import TranscriptView
from karotte.terminal.websocket_handler import listen_to_websocket

# Constants
UPDATE_INTERVAL = 0.1  # seconds


class KarotteApp(App[None]):
    """Main Textual application for karotte dashboard."""

    CSS_PATH = "styles.tcss"  # pyright: ignore[reportUnannotatedClassAttribute]

    BINDINGS = [  # pyright: ignore[reportUnannotatedClassAttribute]
        Binding(key="q", action="quit", description="Quit"),
        Binding(key="f", action="toggle_view", description="Toggle mode"),
        Binding(key="c", action="copy_content", description="Copy markdown"),
    ]

    # Disable Textual's internal logging to avoid terminal pollution
    def on_worker_state_changed(self, event) -> None:  # pyright: ignore[reportMissingParameterType]
        """Suppress worker state change logging."""
        event.stop()

    def __init__(
        self,
        configs: list[EvaluationRunConfig] | None = None,
        transcripts: list[Transcript] | None = None,
        driver_class=None,  # pyright: ignore[reportMissingParameterType]
        css_path=None,  # pyright: ignore[reportMissingParameterType]
        watch_css: bool = False,
        ansi_color: bool = False,
    ):
        # Suppress Textual's console logging
        import logging

        logging.getLogger("textual").setLevel(logging.CRITICAL)

        super().__init__(driver_class, css_path, watch_css, ansi_color)

        # Support two modes: live (from configs) or static (from transcripts)
        if transcripts is not None:
            # Static mode: create configs from transcripts
            self.configs: list[EvaluationRunConfig] = [
                self._config_from_transcript(t, i) for i, t in enumerate(transcripts)
            ]
            self.transcripts: list[Transcript] | None = transcripts
            self.static_mode: bool = True
        elif configs is not None:
            self.configs = configs
            self.transcripts = None
            self.static_mode = False
        else:
            raise ValueError("Must provide either configs or transcripts")

        self.websocket_addresses: list[str] = [
            f"{config.websocket_config.host}:{config.websocket_config.port}"
            for config in self.configs
        ]
        # Event queues for each run
        self.event_queues: list[deque[str]] = [deque() for _ in self.configs]

        # Store all received events per run (for display when switching runs)
        self.received_events: list[list[str]] = [[] for _ in self.configs]
        # Pre-formatted JSON for JSON view mode (full content for copying)
        self.received_events_json: list[list[str]] = [[] for _ in self.configs]
        # Pre-formatted JSON with truncated content for display
        self.received_events_json_truncated: list[list[str]] = [
            [] for _ in self.configs
        ]

        # In static mode, populate event_queues from transcripts
        if self.static_mode and transcripts:
            for i, transcript in enumerate(transcripts):
                for event in transcript.events:
                    self.event_queues[i].append(event.model_dump_json())

        # Track run state per run
        self.run_states: list[RunState | None] = [None for _ in self.configs]

        # Track log file paths per run - set immediately from config (don't wait for events)
        self.run_log_files: list[Path] = [
            Path(f"/tmp/karotte_run_{config.run_id}.log") for config in self.configs
        ]

        # Track log file positions for streaming
        self.log_file_positions: list[int] = [0 for _ in self.configs]

        # References to widgets (set in compose)
        self.run_list: Sidebar | None = None
        self.detail_view: RunView | None = None

        # For live streaming mode
        self.shutdown_events: list[anyio.Event] = []
        self.run_disconnected: list[bool] = [False for _ in self.configs]

        # For capturing main console output
        self.main_console_buffer: StringIO = StringIO()

        # Build/run parameters (set from entrypoints.py for live mode)
        self.runtime: Runtime = "docker"
        self.dev: bool = False
        self.build_context: str = "."
        self.build_secrets: Sequence[str] = ()
        self.cache_from: list[str] | None = None
        self.cache_to: list[str] | None = None
        self.containerized: bool = True
        self.keep_containers: bool = False
        self.mounts: list[str] | None = None
        self.proxy_url: str | None = None

        # Worker tracking for cleanup
        self.build_worker: object | None = None
        self.run_failed: bool = False

        # View mode for transcript display: "json", "logs", "pretty"
        self.view_mode: str = "pretty"

    @override
    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""
        yield Header()

        with Horizontal():
            self.run_list = Sidebar(
                self.configs, initial_index=0, show_main_console=not self.static_mode
            )
            yield self.run_list

            self.detail_view = RunView(self.configs)
            yield self.detail_view

        yield Footer()

    def on_mount(self) -> None:
        """Called when app is mounted."""
        # In live mode, select main console first; in static mode, select first run
        if self.static_mode:
            if self.configs:
                self.select_run(0)
            # Start polling for event queues (both modes use this now)
            self.set_interval(UPDATE_INTERVAL, self.update_transcripts_from_queues)
        else:
            # Live mode: show main console and start build/run worker
            self.select_main_console()
            self.start_build_and_run_worker()
            # Start polling for console output
            self.set_interval(UPDATE_INTERVAL, self.update_main_console)
            # Start animation timer for loading indicators
            self.set_interval(0.1, self.animate_loading_indicators)
            # Start polling for log file updates
            self.set_interval(0.5, self.update_logs_from_files)

    def on_unmount(self) -> None:
        """Called when app is unmounting - cleanup resources."""
        # Cancel the build worker if still running
        if (
            self.build_worker
            and hasattr(self.build_worker, "is_finished")
            and not self.build_worker.is_finished  # pyright: ignore[reportAttributeAccessIssue]
        ):
            if hasattr(self.build_worker, "cancel"):
                self.build_worker.cancel()  # pyright: ignore[reportAttributeAccessIssue]

        # Signal all websocket listeners to shut down
        for event in self.shutdown_events:
            event.set()

    def on_list_view_highlighted(self, event) -> None:  # pyright: ignore[reportMissingParameterType]
        """Handle run highlight in sidebar (arrow keys)."""
        if (
            self.run_list
            and hasattr(event, "list_view")
            and event.list_view == self.run_list
        ):
            highlighted_index = self.run_list.index
            if highlighted_index is not None:
                # Index 0 is main console (in live mode), runs start at 1
                if not self.static_mode and highlighted_index == 0:
                    self.select_main_console()
                else:
                    # Adjust index for runs (subtract 1 if main console is shown)
                    run_index = (
                        highlighted_index - 1
                        if not self.static_mode
                        else highlighted_index
                    )

                    # Only allow navigation to runs that have started (not hidden)
                    if 0 <= run_index < len(self.run_list.run_items):
                        run_item = self.run_list.run_items[run_index]
                        if run_item.status != "hidden":
                            self.select_run(run_index)

    def select_main_console(self) -> None:
        """Select and show the main console view."""
        if self.detail_view:
            self.detail_view.show_main_console = True
            self._set_run_bindings_visible(show=False)

    def select_run(self, i_run: int) -> None:
        """Select a run and update the detail view."""
        if self.detail_view and 0 <= i_run < len(self.configs):
            self.detail_view.show_main_console = False
            self.detail_view.run_index = i_run

            transcript_view = self.detail_view.query_one(
                "#transcript-view", TranscriptView
            )
            transcript_view.clear_transcript()
            transcript_view.set_view_mode(self.view_mode)

            if self.view_mode == "logs":
                # Load log file contents for this run
                self._load_logs_for_run(i_run, from_start=True)
            elif self.view_mode == "json":
                # Load pre-formatted JSON (truncated for display)
                transcript_view.set_json_content(
                    "\n\n".join(self.received_events_json_truncated[i_run])
                )
            else:
                # Replay all received events for this run (pretty mode)
                for event in self.received_events[i_run]:
                    transcript_view.append_event(event)

            # Update header with current state (live mode only)
            if not self.static_mode:
                self._update_run_header(i_run)

            self._set_run_bindings_visible(show=True)

    def start_build_and_run_worker(self) -> None:
        """Start worker to build containers and run evaluations."""
        self.write_to_main_console("Starting build and run process...\n")

        # Start the build/run in a worker
        self.build_worker = self.run_worker(
            self._build_and_run(),
            exclusive=False,
        )

    async def _build_and_run(self) -> None:
        """Worker function that builds and runs containers/processes."""
        try:
            if self.containerized:
                # Create callback that will be called from thread when containers start
                def on_containers_started():
                    # Schedule WebSocket listeners to start in the app's event loop
                    self.call_later(self.start_websocket_listeners)

                # Run in thread pool since it uses subprocess
                await anyio.to_thread.run_sync(  # pyright: ignore[reportAttributeAccessIssue]
                    partial(
                        self._run_containerized_builds_and_runs,
                        self.runtime,
                        self.configs,
                        self.dev,
                        self.build_context,
                        self.keep_containers,
                        self.write_to_main_console,
                        on_containers_started,
                        self.mounts,
                        self.proxy_url,
                    )
                )
            else:
                # Run non-containerized (already async)
                await self._run_non_containerized(
                    self.configs[0],
                    self.write_to_main_console,
                )

                # Start WebSocket listener for non-containerized mode
                self.start_websocket_listeners()

        except Exception:
            self.run_failed = True
            error_msg = f"\n❌ Build/run failed:\n{traceback.format_exc()}\n"
            self.write_to_main_console(error_msg)

    def _build_container(
        self,
        runtime: Runtime,
        build_context: str,
        output_callback: Callable[[str], None] | None = None,
    ):
        """Build the container image."""
        build_command = get_container_build_command(
            runtime,
            build_context,
            cache_from=self.cache_from,
            cache_to=self.cache_to,
            build_secrets=self.build_secrets,
        )

        msg = f"Building container image with command: {shlex.join(build_command)}\n"
        if output_callback:
            output_callback(msg)

        build_result = None

        # Stream output line by line to the callback
        if output_callback:
            process = subprocess.Popen(
                build_command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            if process.stdout:
                for line in process.stdout:
                    output_callback(line)
                build_result_code = process.wait()
            else:
                build_result_code = process.wait()
        else:
            build_result = subprocess.run(
                build_command, check=False, capture_output=True, text=True
            )
            build_result_code = build_result.returncode

        if build_result_code != 0:
            error_msg = "Failed to build environment container image.\n"
            if build_result:
                if build_result.stdout:
                    error_msg += f"Stdout: {build_result.stdout}\n"
                if build_result.stderr:
                    error_msg += f"Stderr: {build_result.stderr}\n"
            if output_callback:
                output_callback(error_msg)
            raise RuntimeError(error_msg)

        success_msg = "Environment container image built successfully.\n"
        if output_callback:
            output_callback(success_msg)

    def _run_containerized_builds_and_runs(
        self,
        runtime: Runtime,
        run_configs: list[EvaluationRunConfig],
        dev: bool,
        build_context: str,
        keep_containers: bool,
        output_callback: Callable[[str], None] | None = None,
        on_containers_started: Callable[[], None] | None = None,
        mounts: list[str] | None = None,
        proxy_url: str | None = None,
    ):
        """Build container and run all configs. Used by TUI worker."""
        # Build container if needed
        if not dev:
            self._build_container(
                runtime, build_context, output_callback=output_callback
            )

        clean_up_old_containers(runtime, [c.run_id for c in run_configs])

        # Start container runs in parallel
        if output_callback:
            output_callback("\n=== Starting container runs in parallel ===\n")
            for config in run_configs:
                output_callback(f"  Container: karotte_run_{config.run_id}\n")
                output_callback(
                    f"  Log file:  /tmp/karotte_run_{config.run_id}.log\n\n"
                )

        # Notify immediately after build so WebSocket listeners can start
        if on_containers_started:
            on_containers_started()

        # Now start the containers - this will block but WebSocket listeners are already running
        from concurrent.futures import ThreadPoolExecutor

        run_containerized_ = partial(
            _run_containerized_worker,
            runtime=runtime,
            dev=dev,
            keep_container=keep_containers,
            build_context=build_context,
            mounts=mounts,
            proxy_url=proxy_url,
        )

        with ThreadPoolExecutor(max_workers=len(run_configs)) as executor:
            list(executor.map(run_containerized_, run_configs))

        if output_callback:
            output_callback("\n=== All runs completed ===\n")
            if keep_containers:
                output_callback("Containers preserved. To copy data:\n")
                output_callback(
                    f"  {get_engine(runtime)} cp karotte_run_{run_configs[0].run_id}:/workdir/ ./out/\n"
                )

    async def _run_non_containerized(
        self,
        run_config: EvaluationRunConfig,
        output_callback: Callable[[str], None] | None = None,
    ):
        """Run directly on host. Used by TUI worker."""
        from karotte.mcp_servers.http_mcp_server import run_server as run_mcp_server

        if output_callback:
            output_callback(f"Starting run {run_config.run_id} on host...\n")

        task = load_task(run_config)

        with run_mcp_server(run_config.mcp_server_config):
            await run_non_containerized(run_config, task)

        if output_callback:
            output_callback(f"Run {run_config.run_id} completed.\n")

    def start_websocket_listeners(self) -> None:
        """Start async tasks to listen to all WebSocket addresses."""
        self.shutdown_events = [anyio.Event() for _ in self.configs]

        for i, address in enumerate(self.websocket_addresses):
            # Create a status callback for this run
            def make_status_callback(run_index: int):
                def callback(status: str):
                    if status == "disconnected":
                        self.run_disconnected[run_index] = True

                    if self.run_list:
                        # Workers run in the same async context, so we can call directly
                        self.run_list.update_run_status(run_index, status)

                    # Also log to main console
                    status_msg = f"Run #{run_index + 1}: {status}"
                    self.write_to_main_console(f"{status_msg}\n")

                return callback

            # Start WebSocket listener in background
            self.run_worker(
                listen_to_websocket(
                    address,
                    self.event_queues[i],
                    self.shutdown_events[i],
                    make_status_callback(i),
                ),
                exclusive=False,
            )

        # Start update timer to poll queues
        self.set_interval(UPDATE_INTERVAL, self.update_transcripts_from_queues)

    def _config_from_transcript(
        self, transcript: Transcript, index: int
    ) -> EvaluationRunConfig:
        """Create an EvaluationRunConfig from a Transcript for display purposes."""
        # Try to extract metadata from TaskStartedEvent
        task_id = transcript.run_id
        model = "unknown"

        for event in transcript.events:
            if isinstance(event, TaskStartedEvent):
                task_id = event.task_id
                if event.model:
                    model = event.model
                break

        return EvaluationRunConfig(
            run_id=transcript.run_id,
            task_id=task_id,
            model=model,
            model_api_key="",
            websocket_config=WebSocketConfig(host="localhost", port=8000 + index),
        )

    def update_transcripts_from_queues(self) -> None:
        """Poll event queues and update run states for all runs."""
        if self.detail_view is None:
            return

        selected_index = self.detail_view.run_index

        # Process events from ALL queues, not just the selected one
        for run_index, queue in enumerate(self.event_queues):
            while queue:
                try:
                    event_json = queue.popleft()
                    # Store event for this run
                    self.received_events[run_index].append(event_json)

                    # If this is the selected run, display in transcript view (unless in logs mode)
                    if (
                        run_index == selected_index
                        and selected_index is not None
                        and self.view_mode != "logs"
                    ):
                        try:
                            transcript_view = self.detail_view.query_one(
                                "#transcript-view", TranscriptView
                            )
                            transcript_view.append_event(event_json)
                        except NoMatches:
                            # Widget not mounted yet; event is already stored in
                            # received_events and will be displayed when the view loads
                            pass

                    # Show run in sidebar on first event (but don't overwrite completed/failed/error status)
                    if len(self.received_events[run_index]) == 1 and self.run_list:
                        current_status = self.run_list.run_items[run_index].status
                        if current_status in ("hidden", "connecting"):
                            self.run_list.update_run_status(run_index, "connected")

                    # Update run state
                    try:
                        event = TypeAdapter(Event).validate_json(event_json)

                        # Store formatted JSON (skip streaming events)
                        if not isinstance(
                            event, (MessageChunkEvent, MessageChunkResetEvent)
                        ):
                            formatted_json = event.model_dump_json(indent=2)
                            self.received_events_json[run_index].append(formatted_json)

                            # Also store simplified version for display (images replaced with placeholder)
                            if isinstance(
                                event, (ToolCallCompletedEvent, MessageAddedEvent)
                            ):
                                from karotte.terminal.transcript_view import (
                                    simplify_event_for_display,
                                )

                                simplified_event = simplify_event_for_display(event)
                                simplified_formatted = simplified_event.model_dump_json(
                                    indent=2
                                )
                            else:
                                simplified_formatted = formatted_json
                            self.received_events_json_truncated[run_index].append(
                                simplified_formatted
                            )

                        if self.run_states[run_index] is None:
                            if isinstance(event, TaskStartedEvent):
                                self.run_states[run_index] = RunState(event)
                        else:
                            run_state = self.run_states[run_index]
                            if run_state is not None:
                                run_state.apply(event)

                        # Update header with new state if this is the selected run
                        if (
                            run_index == selected_index
                            and self.run_states[run_index] is not None
                        ):
                            self._update_run_header(run_index)

                        # Update navbar color for this run based on run_state or event
                        if self.run_list:
                            if self.run_states[run_index] is not None:
                                run_state = self.run_states[run_index]
                                if run_state is not None:
                                    self.run_list.update_run_color(
                                        run_index, run_state.status
                                    )
                            elif isinstance(event, TaskCompletedEvent):
                                # Handle TaskCompletedEvent even without RunState
                                # (e.g., when error occurs before TaskStartedEvent)
                                self.run_list.update_run_color(run_index, event.status)
                    except Exception:
                        pass

                except IndexError:
                    break

            # After draining the queue: if the websocket disconnected and the
            # run never reached a terminal state, mark it as an error so the
            # sidebar doesn't spin forever.
            if self.run_disconnected[run_index] and not queue:
                run_state = self.run_states[run_index]
                if run_state is None or not run_state.is_terminal:
                    if self.run_list:
                        self.run_list.update_run_color(run_index, "error")

    def animate_loading_indicators(self) -> None:
        """Animate the loading indicators for connecting runs."""
        if self.run_list:
            for item in self.run_list.run_items:
                item.animate_loading()

    def _update_run_header(self, run_index: int) -> None:
        """Update the run header with current state."""
        if not self.detail_view or self.detail_view.run_index != run_index:
            return

        run_state = self.run_states[run_index]
        if run_state is None:
            return

        header = self.detail_view.query_one("#run-header", RunHeader)
        header.update_from_run_state(run_state, self.view_mode)

    def update_main_console(self) -> None:
        """Update main console view with captured stdout/stderr."""
        if not self.detail_view:
            return

        # Check if there's new output in the buffer
        new_output = self.main_console_buffer.getvalue()
        if new_output:
            try:
                main_console = self.detail_view.query_one(
                    "#main-console-view", ConsoleView
                )
            except Exception:
                # Widget not mounted yet; buffer is preserved and will be
                # processed on the next timer tick
                return
            # Split by lines and append each
            for line in new_output.split("\n"):
                if line:  # Skip empty lines
                    main_console.append_line(line)
            # Clear the buffer after reading
            self.main_console_buffer.truncate(0)
            self.main_console_buffer.seek(0)

    def write_to_main_console(self, text: str) -> None:
        """Write text to the main console buffer."""
        self.main_console_buffer.write(text)

    def action_toggle_view(self) -> None:
        """Cycle between JSON, Logs, and Pretty views."""
        # Only allow toggle if a run is selected
        if not self.detail_view or self.detail_view.run_index is None:
            return

        # Cycle through modes: pretty -> json -> logs -> pretty
        if self.view_mode == "pretty":
            self.view_mode = "json"
        elif self.view_mode == "json":
            self.view_mode = "logs"
        else:
            self.view_mode = "pretty"

        # Refresh the current transcript view
        run_index = self.detail_view.run_index
        if 0 <= run_index < len(self.received_events):
            transcript_view = self.detail_view.query_one(
                "#transcript-view", TranscriptView
            )
            transcript_view.clear_transcript()
            transcript_view.set_view_mode(self.view_mode)

            if self.view_mode == "logs":
                # Load and display log file contents from beginning
                self._load_logs_for_run(run_index, from_start=True)
            elif self.view_mode == "json":
                # Load pre-formatted JSON (truncated for display)
                transcript_view.set_json_content(
                    "\n\n".join(self.received_events_json_truncated[run_index])
                )
            else:
                # Reload all events in pretty mode
                for event in self.received_events[run_index]:
                    transcript_view.append_event(event)

        # Update the copy binding description based on current mode
        self._update_bindings_for_view_mode()

    def _update_bindings_for_view_mode(self) -> None:
        """Update binding descriptions based on current view mode."""
        from textual.binding import Binding

        # Update copy binding description
        copy_description_map = {
            "pretty": "Copy markdown",
            "json": "Copy JSON",
            "logs": "Copy logs",
        }
        copy_description = copy_description_map.get(self.view_mode, "Copy markdown")

        # Replace the binding for 'c' key with updated description
        # We need to clear and re-add because bind() appends rather than replaces
        self._bindings.key_to_bindings["c"] = [
            Binding(
                key="c", action="copy_content", description=copy_description, show=True
            )
        ]
        self.refresh_bindings()

    def _set_run_bindings_visible(self, *, show: bool) -> None:
        """Show or hide bindings that only apply when a run is selected."""
        from textual.binding import Binding

        copy_description_map = {
            "pretty": "Copy markdown",
            "json": "Copy JSON",
            "logs": "Copy logs",
        }
        copy_description = copy_description_map.get(self.view_mode, "Copy markdown")

        self._bindings.key_to_bindings["f"] = [
            Binding(key="f", action="toggle_view", description="Toggle mode", show=show)
        ]
        self._bindings.key_to_bindings["c"] = [
            Binding(
                key="c", action="copy_content", description=copy_description, show=show
            )
        ]
        self.refresh_bindings()

    def action_copy_content(self) -> None:
        """Copy content to clipboard based on current view mode."""
        if not self.detail_view or self.detail_view.run_index is None:
            self.notify("No run selected", severity="warning")
            return

        run_index = self.detail_view.run_index
        if not (0 <= run_index < len(self.received_events)):
            self.notify("Invalid run index", severity="error")
            return

        if self.view_mode == "pretty":
            content = self._generate_markdown_for_run(run_index)
            content_type = "markdown"
        elif self.view_mode == "json":
            content = self._get_transcript_json_for_run(run_index)
            content_type = "JSON"
        else:  # logs
            transcript_view = self.detail_view.query_one(
                "#transcript-view", TranscriptView
            )
            from textual.widgets import TextArea

            logs_area = transcript_view.query_one("#transcript-logs", TextArea)
            content = logs_area.text
            content_type = "logs"

            if content == "Waiting for connection...":
                content = ""

        if not content:
            self.notify("No content to copy", severity="warning")
            return

        self.copy_to_clipboard(content)
        self.notify(f"Copied {content_type} to clipboard")

    def _generate_markdown_for_run(self, run_index: int) -> str:
        """Generate full markdown content for a run from stored events."""
        from pydantic import TypeAdapter

        from karotte.schemas.transcript import (
            Event,
            MessageChunkEvent,
            MessageChunkResetEvent,
        )
        from karotte.terminal.transcript_view import TranscriptView

        markdown_parts: list[str] = []
        # Create a temporary TranscriptView just to use its conversion method
        temp_view = TranscriptView()

        for event_json in self.received_events[run_index]:
            try:
                event = TypeAdapter(Event).validate_json(event_json)
                if isinstance(event, (MessageChunkEvent, MessageChunkResetEvent)):
                    continue
                md = temp_view.convert_event_to_markdown(event)
                if md and md.strip():
                    markdown_parts.append(md.strip())
            except Exception:
                pass

        return "\n\n".join(markdown_parts)

    def _get_transcript_json_for_run(self, run_index: int) -> str:
        """Get the full transcript as JSON for a run."""
        # In static mode, use the stored transcript directly
        if self.static_mode and self.transcripts:
            return self.transcripts[run_index].model_dump_json(indent=2)

        # In live mode, build a transcript from accumulated events
        from pydantic import TypeAdapter

        from karotte.schemas.transcript import (
            Event,
            MessageChunkEvent,
            MessageChunkResetEvent,
        )

        events: list[Event] = []
        for event_json in self.received_events[run_index]:
            try:
                event = TypeAdapter(Event).validate_json(event_json)
                if not isinstance(event, (MessageChunkEvent, MessageChunkResetEvent)):
                    events.append(event)
            except Exception:
                pass

        transcript = Transcript(
            run_id=self.configs[run_index].run_id,
            events=events,
        )
        return transcript.model_dump_json(indent=2)

    def _load_logs_for_run(self, run_index: int, from_start: bool = False) -> None:
        """Load and display the log file for a specific run."""
        if not self.detail_view:
            return

        log_file = self.run_log_files[run_index]
        transcript_view = self.detail_view.query_one("#transcript-view", TranscriptView)

        if log_file.exists():
            try:
                with open(log_file) as f:
                    if from_start:
                        # Load from beginning and reset position
                        self.log_file_positions[run_index] = 0
                        log_contents = f.read()
                        self.log_file_positions[run_index] = f.tell()
                    else:
                        # Load only new content since last read
                        f.seek(self.log_file_positions[run_index])
                        log_contents = f.read()
                        self.log_file_positions[run_index] = f.tell()

                    if log_contents:
                        transcript_view.append_event(log_contents)
                    elif from_start:
                        transcript_view.append_event("Log file is empty")
            except Exception as e:
                transcript_view.append_event(f"Error reading log file: {e}")
        elif from_start:
            transcript_view.append_event("Log file not available yet")

    def update_logs_from_files(self) -> None:
        """Poll log files for new content and update transcript if in logs mode."""
        if self.view_mode != "logs" or not self.detail_view:
            return

        selected_index = self.detail_view.run_index
        if selected_index is None or not (
            0 <= selected_index < len(self.run_log_files)
        ):
            return

        # Stream new log content for the selected run
        self._load_logs_for_run(selected_index, from_start=False)


def run_static_dashboard(transcript_dir: Path) -> None:
    """Run the dashboard in static mode with transcripts from a directory."""

    if not transcript_dir.exists():
        logger.error("Directory {} does not exist", transcript_dir)
        return

    transcripts = discover_transcripts(transcript_dir)

    if not transcripts:
        logger.error("No transcripts (*.json) found in {}", transcript_dir)
        return

    transcripts = sorted(transcripts, key=lambda t: t.run_id)
    logger.info("Loaded {} transcript(s)", len(transcripts))

    app = KarotteApp(transcripts=transcripts)
    app.run()


def _run_containerized_worker(
    run_config: EvaluationRunConfig,
    runtime: Runtime,
    dev: bool,
    keep_container: bool,
    build_context: str = ".",
    mounts: list[str] | None = None,
    proxy_url: str | None = None,
):
    """Run a single containerized evaluation run. Must be a module-level function for pickling."""
    run_command, _ = get_container_run_command(
        run_config, runtime, dev, keep_container, build_context, mounts, proxy_url
    )

    # Write output to log file that can be tailed
    log_file = Path(f"/tmp/karotte_run_{run_config.run_id}.log")
    with open(log_file, "w") as f:
        subprocess.run(run_command, check=True, stdout=f, stderr=subprocess.STDOUT)


if __name__ == "__main__":
    # Test with dummy configs
    test_configs = [
        EvaluationRunConfig(
            run_id=f"run_{i}",
            task_id=f"task_{i}",
            model="gpt-4" if i % 2 == 0 else "claude-3",
            model_api_key="test_key",
            websocket_config=WebSocketConfig(host="localhost", port=8000 + i),
        )
        for i in range(4)
    ]

    app = KarotteApp(configs=test_configs)
    app.run()
