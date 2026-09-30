from karotte import EvaluationRunConfig
from karotte.schemas import HttpMcpServerConfig, Message


def test_fake_model_interface():
    """Verify this environment provides a valid fake_model module.

    This is a smoke test that checks the interface exists and has the right signature.
    It does not test specific message content since that's expected to be customized.
    """
    from environment.fake_model import get_messages

    # Test it's callable with the right signature
    config = EvaluationRunConfig(
        run_id="test",
        task_id="test-task",
        model="test-model",
        model_api_key="test-key",
        mcp_server_config=HttpMcpServerConfig(host="0.0.0.0", port=8080),
        transcript_file="transcript.json",
    )
    messages = get_messages(config)

    # Verify return type without checking specific content
    assert isinstance(messages, list), "get_messages should return a list"
    assert all(isinstance(m, Message) for m in messages), (
        "All items should be Message instances"
    )
    # Don't assert specific content - that's user-customizable
