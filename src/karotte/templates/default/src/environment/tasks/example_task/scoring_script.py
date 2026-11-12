import json
import re
import sys
from pathlib import Path

if __name__ == "__main__":
    submission_path, output_path = sys.argv[1], sys.argv[2]

    answer = Path(submission_path).read_text().strip()

    # Match the whole answer. A substring check would also accept "3.12.110"
    # or a list of guesses such as "3.10 3.11 3.12.11 3.13".
    score = 1.0 if re.fullmatch(r"(Python )?3\.12\.11", answer) else 0.0

    Path(output_path).write_text(json.dumps({"score": score, "metadata": {}}))
