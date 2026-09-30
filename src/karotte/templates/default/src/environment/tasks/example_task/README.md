## Example Task

### Background

A minimal multi-step task for testing the evaluation harness.
The student must discover the Python executable path and then query its version.

### Motivation

Validates that the student can chain information across steps and use bash for system introspection.

## Data

No external data.

## Scoring

Step 1: Regex match on `/workdir/.venv/bin/python`.
Step 2: Scoring script to ensure that the student has written the Python version to the specified file.

### Hints

Step 1: "A single bash tool call will suffice."
Step 2: "Use the python executable from your previous answer."
