"""Check that a transcript completed with a passing status."""

import sys
from pathlib import Path

from karotte.schemas.transcript import TaskCompletedEvent, Transcript


def main():
    transcript_path = Path(sys.argv[1])
    t = Transcript.model_validate_json(transcript_path.read_text())

    completed = [e for e in t.events if isinstance(e, TaskCompletedEvent)]
    if not completed:
        print("FAIL: No TaskCompletedEvent found in transcript")
        sys.exit(1)

    status = completed[-1].status
    if status != "passed":
        print(f"FAIL: Run completed with status: {status}")
        sys.exit(1)

    print("OK: Run passed")


if __name__ == "__main__":
    main()
