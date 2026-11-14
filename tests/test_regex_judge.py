import ast
import re
from pathlib import Path
from typing import Any

import pytest

from karotte import EvaluationRunConfig, Step, Task
from karotte.judges.judge import Judge
from karotte.judges.regex_judge import RegexJudge
from karotte.schemas.chat import ChatCompletionMessageToolCall, Function, Message, Role
from karotte.schemas.transcript import (
    MessageAddedEvent,
    StepStartedEvent,
    Transcript,
)

DOCS = Path(__file__).parent.parent / "docs" / "tasks"


def test_general_value_match(transcript: Transcript):
    expected_results = RegexJudge([re.compile(r".*23")])

    transcript.events.append(MessageAddedEvent(message=Message(content="203049823984")))

    scoring = expected_results.evaluate(transcript)

    assert scoring.score == 1
    assert scoring.metadata == {}
    assert scoring.continue_task is True


@pytest.mark.parametrize("role", ["user", "system", "tool", "function"])
@pytest.mark.parametrize(
    "content", ["answer: 42", [{"type": "text", "text": "answer: 42"}]]
)
def test_non_assistant_messages_do_not_match(
    transcript: Transcript, role: Role, content: str | list[dict[str, str]]
):
    transcript.events.extend(
        [
            MessageAddedEvent(message=Message(role=role, content=content)),
            MessageAddedEvent(message=Message(content="answer: 41")),
        ]
    )

    scoring = RegexJudge([re.compile("answer: 42")]).evaluate(transcript)

    assert scoring.score == 0
    assert scoring.metadata == {"answer: 42": "Transcript contains no match."}
    assert scoring.continue_task is False


@pytest.mark.parametrize(
    "message",
    [
        Message(content=None),
        Message(content=""),
        Message(content=[]),
        Message(content=None, reasoning_content="answer: 42"),
        Message(
            content=None,
            tool_calls=[
                ChatCompletionMessageToolCall(
                    id="call_1",
                    type="function",
                    function=Function(
                        name="bash", arguments='{"command": "answer: 42"}'
                    ),
                )
            ],
        ),
        Message(content=[{"type": "image_url", "image_url": {"url": "answer: 42"}}]),
        Message(content=[{"type": "text", "text": "wrong", "extra": "answer: 42"}]),
        Message(content=[{"type": "text", "text": None, "extra": "answer: 42"}]),
        Message(content=[{"type": "text", "text": 42}]),
    ],
)
def test_only_assistant_text_is_matched(transcript: Transcript, message: Message):
    transcript.events.append(MessageAddedEvent(message=message))

    scoring = RegexJudge([re.compile("42")]).evaluate(transcript)

    assert scoring.score == 0
    assert scoring.continue_task is False


def test_multipart_assistant_text_matches(transcript: Transcript):
    transcript.events.append(
        MessageAddedEvent(
            message=Message(
                content=[
                    {"type": "text", "text": "answer:"},
                    {"type": "image_url", "image_url": {"url": "image.png"}},
                    {"type": "text", "text": "42"},
                ]
            )
        )
    )

    scoring = RegexJudge([re.compile(r"answer:\s+42")]).evaluate(transcript)

    assert scoring.score == 1
    assert scoring.metadata == {}
    assert scoring.continue_task is True


def test_assistant_text_from_earlier_steps_is_included(transcript: Transcript):
    transcript.events.extend(
        [
            StepStartedEvent(step=0),
            MessageAddedEvent(message=Message(content="answer:")),
            StepStartedEvent(step=1),
            MessageAddedEvent(message=Message(role="user", content="next step")),
            MessageAddedEvent(message=Message(content="42")),
        ]
    )

    scoring = RegexJudge([re.compile(r"answer:\s+42")]).evaluate(transcript)

    assert scoring.score == 1
    assert scoring.continue_task is True


def test_empty_pattern_list_passes_without_assistant_text(transcript: Transcript):
    transcript.events.append(
        MessageAddedEvent(message=Message(role="user", content="answer: 42"))
    )

    scoring = RegexJudge([]).evaluate(transcript)

    assert scoring.score == 1
    assert scoring.metadata == {}
    assert scoring.continue_task is True


def test_general_value_mismatch(transcript: Transcript):
    expected_results = RegexJudge([re.compile(r"123")])

    transcript.events.append(MessageAddedEvent(message=Message(content="203049823984")))

    scoring = expected_results.evaluate(transcript)

    assert scoring.score == 0
    assert scoring.metadata == {"123": "Transcript contains no match."}
    assert scoring.continue_task is False


def test_general_value_mismatch_for_mutliple_values(transcript: Transcript):
    expected_results = RegexJudge(
        [re.compile(r"123"), re.compile(r"456"), re.compile(r"20304")]
    )

    transcript.events.append(MessageAddedEvent(message=Message(content="203049823984")))

    scoring = expected_results.evaluate(transcript)

    assert scoring.score == 0
    assert scoring.metadata == {
        "123": "Transcript contains no match.",
        "456": "Transcript contains no match.",
    }
    assert scoring.continue_task is False


def test_general_value_match_multiline(transcript: Transcript):
    expected_results = RegexJudge([re.compile(r".*2.*3", re.DOTALL)])

    transcript.events.append(MessageAddedEvent(message=Message(content="\n2\n03\n")))

    scoring = expected_results.evaluate(transcript)

    assert scoring.score == 1
    assert scoring.metadata == {}
    assert scoring.continue_task is True


# The examples in the docs anchor their patterns to the end of the assistant
# text. These tests run the docs' code blocks, so they check the patterns as
# the docs show them.


def _run_docs_example(page: str, containing: str) -> dict[str, Any]:
    """Run the Python block on `page` that contains `containing`.

    Returns the block's namespace. A final bare expression is stored as `_`.
    """
    blocks = re.findall(
        r"^```python\n(.*?)^```$",
        (DOCS / page).read_text(),
        re.MULTILINE | re.DOTALL,
    )
    [block] = [b for b in blocks if containing in b]
    tree = ast.parse(block)
    last = tree.body[-1]
    if isinstance(last, ast.Expr):
        tree.body[-1] = ast.Assign([ast.Name("_", ast.Store())], last.value)
        ast.fix_missing_locations(tree)
    namespace: dict[str, Any] = {"__name__": "docs_example"}
    exec(compile(tree, page, "exec"), namespace)
    return namespace


def _score(judge: Judge, transcript: Transcript, replies: list[str]) -> float:
    transcript.events.extend(
        MessageAddedEvent(message=Message(content=reply)) for reply in replies
    )
    return judge.evaluate(transcript).score


GUESS_LIST = "\n".join(f"answer: {n}" for n in range(10_000))


@pytest.fixture
def create_task_example(sample_config: EvaluationRunConfig) -> Step:
    """The step of the `add-small` task (2 + 3) in creating-tasks-in-code.md."""
    namespace = _run_docs_example("creating-tasks-in-code.md", "create_task(")
    task: type[Task] = namespace["add_small"]
    [step] = task(sample_config).steps
    return step


def test_create_task_example_asks_to_end_with_the_answer(create_task_example: Step):
    assert create_task_example.instructions == (
        "What is 2 + 3? End your reply with `answer: <number>`."
    )


@pytest.mark.parametrize(
    "replies",
    [
        ["answer: 5"],
        ["answer: 5."],
        ["answer: 5\n"],
        ["2 + 3 adds up to 5.\n\nanswer: 5"],
        ["Let me add them.", "answer: 5"],
    ],
)
def test_create_task_example_accepts_a_reply_ending_with_the_answer(
    create_task_example: Step, transcript: Transcript, replies: list[str]
):
    assert _score(create_task_example.judge, transcript, replies) == 1


@pytest.mark.parametrize(
    "replies",
    [
        pytest.param(["answer: 5, 6, 7"], id="list"),
        pytest.param(["answer: 5 or 7"], id="or"),
        pytest.param(["answer: 5-7"], id="range"),
        pytest.param(["answer: 5", "Actually, Answer: 7"], id="changed"),
        pytest.param(["answer: 5.5"], id="decimal"),
        pytest.param(["answer: 55"], id="longer"),
        pytest.param([GUESS_LIST], id="guess-list"),
        pytest.param(["answer: 6"], id="wrong"),
    ],
)
def test_create_task_example_rejects_anything_but_one_final_answer(
    create_task_example: Step, transcript: Transcript, replies: list[str]
):
    assert _score(create_task_example.judge, transcript, replies) == 0


@pytest.mark.parametrize(
    ("replies", "score"),
    [
        (["answer: 42"], 1),
        (["It is 42.\n\nanswer: 42."], 1),
        (["answer: 42 or 43"], 0),
        (["answer: 42", "Actually, answer: 43"], 0),
        (["answer: 420"], 0),
    ],
)
def test_scoring_example(transcript: Transcript, replies: list[str], score: float):
    namespace = _run_docs_example("scoring.md", "from karotte.judges import RegexJudge")
    judge: Judge = namespace["_"]

    assert _score(judge, transcript, replies) == score


@pytest.mark.parametrize(
    ("replies", "score"),
    [
        (["path: /usr/bin/python3"], 1),
        (["`which python3` prints it.\n\npath: /usr/bin/python3."], 1),
        (["path: /a /b /workdir/.venv/bin/python"], 0),
        (["path: /usr/bin/python3\npath: /usr/bin/perl"], 0),
        (["path: /usr/bin/python3", "Actually, path: /usr/bin/perl"], 0),
    ],
)
def test_tasks_and_steps_example(
    sample_config: EvaluationRunConfig,
    transcript: Transcript,
    replies: list[str],
    score: float,
):
    namespace = _run_docs_example("tasks-and-steps.md", "class FindPythonStep")
    step = namespace["FindPythonStep"](config=sample_config)

    assert step.instructions.endswith("End your reply with `path: <path>`.")
    assert _score(step.judge, transcript, replies) == score
