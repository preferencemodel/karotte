from pathlib import Path

from loguru import logger

from karotte.schemas.transcript import Transcript


def discover_transcripts(transcript_dir: Path) -> list[Transcript]:
    transcripts: list[Transcript] = []

    for file in (f for f in transcript_dir.glob("*.json") if f.is_file()):
        try:
            transcript = Transcript.model_validate_json(file.read_text())
            transcripts.append(transcript)
        except Exception as e:
            logger.warning("Error parsing {}: {}", file, e)

    return transcripts
