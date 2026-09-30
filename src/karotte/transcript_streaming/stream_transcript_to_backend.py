from collections.abc import AsyncIterator

import httpx
from karotte.backend_client import BackendClient
from karotte.schemas.evaluation_run_config import EvaluationRunConfig
from karotte.schemas.run_state import RunState
from karotte.schemas.transcript import (
    Event,
    MessageChunkEvent,
    MessageChunkResetEvent,
    TaskCompletedEvent,
    TaskStartedEvent,
    TokenUsageEvent,
)
from loguru import logger


async def stream_transcript_to_backend(
    events: AsyncIterator[Event], run_config: EvaluationRunConfig
):
    if run_config.backend_uri is None:
        raise ValueError("backend_uri is required to stream transcripts to the backend")

    run_state: RunState | None = None

    async with BackendClient(run_config.backend_uri) as client:
        initialized = False
        seq = 0

        async for event in events:
            if isinstance(event, (MessageChunkEvent, MessageChunkResetEvent)):
                continue

            if not initialized:
                assert isinstance(event, TaskStartedEvent)
                run_state = RunState(event)
                try:
                    # A reconnected run replays its events from the start, so
                    # always number from seq 0 and let the positional write
                    # overwrite existing slots in place. Only skip re-creating
                    # the transcript when one already exists (a fresh run 404s).
                    length = await client.get_transcript_length(run_config.run_id)
                    if length is None:
                        await client.create_transcript(run_config.run_id)
                    seq = 0
                except httpx.HTTPError:
                    logger.exception(
                        f"Failed to initialize transcript for run {run_config.run_id}"
                    )
                    raise
                initialized = True

            try:
                await client.append_transcript(
                    run_config.run_id, event.model_dump(mode="json"), seq=seq
                )
            except httpx.HTTPError:
                logger.exception(
                    f"Failed to append event to backend transcript for run {run_config.run_id}"
                )
                raise
            seq += 1

            assert run_state is not None
            run_state.apply(event)

            if isinstance(
                event, TaskStartedEvent | TaskCompletedEvent | TokenUsageEvent
            ):
                try:
                    await client.update_run_state(
                        run_config.run_id,
                        status=run_state.status,
                        score=run_state.score,
                        input_tokens=run_state.total_input_tokens,
                        output_tokens=run_state.total_output_tokens,
                        cache_read_tokens=run_state.total_cache_read_tokens,
                        cache_write_tokens=run_state.total_cache_write_tokens,
                    )
                except httpx.HTTPError:
                    logger.exception(
                        f"Failed to update run state for run {run_config.run_id}"
                    )
                    raise
