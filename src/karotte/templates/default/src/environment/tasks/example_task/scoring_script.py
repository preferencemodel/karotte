import json
import sys
from pathlib import Path

if __name__ == "__main__":
    submission_path, output_path = sys.argv[1], sys.argv[2]

    answer = Path(submission_path).read_text().strip()

    score = 1.0 if "3.12.11" in answer else 0.0

    Path(output_path).write_text(json.dumps({"score": score, "metadata": {}}))
