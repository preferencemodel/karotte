# Connecting a backend

Without a backend, Karotte writes the transcript to a file and saves artifacts locally (see [Artifacts and transcripts](../tasks/artifacts-and-transcripts.md)).
To collect runs in one place, set `backend_uri` in the run config (see [Run config](../running/run-config.md)):

```json
"backend_uri": "https://runs.example.com"
```

Karotte then streams each run's transcript to the backend, reports the run's state, and uploads artifacts through it.

## Endpoints

Karotte calls these HTTP endpoints on `backend_uri`.
Each request passes `run_id` as a query parameter, except presign, which sends it in the body.

| Request                                | Purpose                                                                                                                                                  |
| -------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `POST /api/internal/create_transcript` | Starts a transcript.                                                                                                                                     |
| `GET /api/internal/transcript_length`  | Returns the number of stored events as a JSON number, or 404 if there are none.                                                                          |
| `POST /api/internal/append_transcript` | Sends `{"event": ..., "seq": n}` in the body. If Karotte writes the same `seq` again, the backend must overwrite the stored event, not append a new one. |
| `POST /api/internal/update_run_state`  | Sends `status` (`running`, `passed`, `failed`, `error`), `score` and token counts in the body.                                                           |
| `POST /api/artifacts/presign`          | Sends `{"run_id", "artifact_paths"}` in the body and returns `{"presigned_urls": {path: url}}`. Karotte then PUTs each file gzipped.                     |

### Transcripts

When a run starts, Karotte asks for the transcript length.
If it gets a 404, it creates the transcript.
Otherwise it reuses the existing one.
It then appends every event, with `seq` counting up from 0.
When a run reconnects, it replays its events from the start.
That's why a repeated `seq` must overwrite the stored event.

Karotte doesn't send streaming message chunks (`message_chunk`, `message_chunk_reset`).

### Run state

Karotte posts `update_run_state` after the task starts, after each token usage event, and when the task completes:

```json
{
    "status": "running",
    "score": null,
    "input_tokens": 1200,
    "output_tokens": 340,
    "cache_read_tokens": null,
    "cache_write_tokens": null
}
```

Token counts are totals for the run so far.
They're `null` until they're known.

### Artifacts

For each artifact, Karotte asks `presign` for one URL per file.
It then `PUT`s each file gzip-compressed, with `Content-Encoding: gzip`.
For a directory artifact, each file's path is `<directory name>/<path inside it>`.
If an upload fails, Karotte logs the failure, and the run doesn't fail because of it.

## Retries

Karotte retries `/api/internal/` requests on connection errors, timeouts and 5xx responses.
It uses exponential backoff and keeps trying for up to 25 minutes per request.
Each attempt times out after 30 seconds.
Other error responses aren't retried.
`presign` is retried the same way, but for up to 5 minutes.

## Authentication

If the file at `KAROTTE_BACKEND_TOKEN_PATH` (default `/var/run/secrets/service-account-token`) exists, requests carry `Authorization: Bearer <token>`.
Karotte reads the file again for every request, so the token can rotate during a run.
If the file doesn't exist, requests carry no `Authorization` header.

## External agent

If you set `"agent": "external"` in the run config, the backend also supplies the model's messages.
Karotte still runs the steps and executes the tool calls.

| Request                             | Purpose                                                                          |
| ----------------------------------- | -------------------------------------------------------------------------------- |
| `GET /api/internal/get_message`     | Returns the next message, or 404 while there isn't one.                          |
| `POST /api/internal/delete_message` | Consumes the message that was just returned. The response must have a JSON body. |

`get_message` returns a `message_added` event.
Karotte treats it as a long poll: after a 404, it calls again right away.
If no message arrives within 120 minutes, the run fails with a timeout.
The external agent only works when `backend_uri` is set.

## Event schema

The events are the models in [`karotte.schemas.transcript`](https://github.com/preferencemodel/karotte/blob/main/src/karotte/schemas/transcript.py), serialized as JSON.
Each event Karotte sends has a `type`, such as `task_started`, `message_added`, `scoring` or `task_completed`.
