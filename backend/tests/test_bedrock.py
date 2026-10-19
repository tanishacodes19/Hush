"""Unit tests for BedrockService with mocked AWS responses and errors.

Tests MUST NOT make real AWS calls.
"""

from unittest.mock import MagicMock
import pytest
from botocore.exceptions import ClientError, EndpointConnectionError

from app.schemas import Message
from app.services.bedrock import BedrockConfigError, BedrockService, BedrockServiceError


def make_mock_converse_response(text: str) -> dict:
    """Helper to generate a valid Bedrock Converse API response structure."""
    return {
        "output": {
            "message": {
                "role": "assistant",
                "content": [{"text": text}],
            }
        },
        "stopReason": "end_turn",
        "usage": {
            "inputTokens": 45,
            "outputTokens": 20,
            "totalTokens": 65,
        },
    }


def test_missing_region_raises_config_error():
    """Service must raise BedrockConfigError when region is not configured."""
    service = BedrockService(region=None, model_id="test-model")
    with pytest.raises(BedrockConfigError, match="AWS region is not configured"):
        service.generate_reply([Message(speaker="alex", text="hello")])


def test_missing_model_id_raises_config_error():
    """Service must raise BedrockConfigError when model_id is not configured."""
    service = BedrockService(region="us-east-1", model_id=None)
    with pytest.raises(BedrockConfigError, match="Bedrock model ID is not configured"):
        service.generate_reply([Message(speaker="alex", text="hello")])


def test_generate_reply_calls_converse_with_expected_payload():
    """generate_reply calls client.converse and returns extracted text."""
    mock_client = MagicMock()
    mock_client.converse.return_value = make_mock_converse_response(
        "Here is a helpful response to the team discussion."
    )

    service = BedrockService(
        region="us-east-1",
        model_id="anthropic.claude-test-v1",
        client=mock_client,
    )

    context = [
        Message(speaker="alex", text="The build is broken again."),
        Message(speaker="sam", text="Which branch?"),
    ]

    result = service.generate_reply(context)

    assert result == "Here is a helpful response to the team discussion."
    mock_client.converse.assert_called_once()
    call_kwargs = mock_client.converse.call_args.kwargs
    assert call_kwargs["modelId"] == "anthropic.claude-test-v1"
    assert "alex: The build is broken again." in call_kwargs["messages"][0]["content"][0]["text"]
    assert "sam: Which branch?" in call_kwargs["messages"][0]["content"][0]["text"]
    assert call_kwargs["inferenceConfig"]["maxTokens"] == 512


def test_generate_clarification_calls_converse_with_expected_payload():
    """generate_clarification calls client.converse with focused prompt."""
    mock_client = MagicMock()
    mock_client.converse.return_value = make_mock_converse_response(
        "Which specific service threw the 500 status code?"
    )

    service = BedrockService(
        region="us-east-1",
        model_id="anthropic.claude-test-v1",
        client=mock_client,
    )

    context = [Message(speaker="alex", text="We are getting 500 errors.")]
    result = service.generate_clarification(context)

    assert result == "Which specific service threw the 500 status code?"
    mock_client.converse.assert_called_once()
    call_kwargs = mock_client.converse.call_args.kwargs
    assert call_kwargs["modelId"] == "anthropic.claude-test-v1"
    assert call_kwargs["inferenceConfig"]["maxTokens"] == 150
    assert "clarifying question" in call_kwargs["system"][0]["text"]


def test_client_error_wrapped_into_bedrock_service_error():
    """AWS ClientError is caught and raised as BedrockServiceError."""
    mock_client = MagicMock()
    error_response = {
        "Error": {
            "Code": "ValidationException",
            "Message": "Invalid model identifier",
        }
    }
    mock_client.converse.side_effect = ClientError(error_response, "Converse")

    service = BedrockService(
        region="us-east-1",
        model_id="invalid-model-id",
        client=mock_client,
    )

    with pytest.raises(BedrockServiceError, match="Amazon Bedrock Converse invocation failed"):
        service.generate_reply([Message(speaker="alex", text="Hello")])


def test_network_connection_error_wrapped():
    """Network connection failure is wrapped into BedrockServiceError."""
    mock_client = MagicMock()
    mock_client.converse.side_effect = EndpointConnectionError(endpoint_url="https://bedrock.test")

    service = BedrockService(
        region="us-east-1",
        model_id="test-model",
        client=mock_client,
    )

    with pytest.raises(BedrockServiceError, match="Amazon Bedrock Converse invocation failed"):
        service.generate_reply([Message(speaker="alex", text="Hello")])


def test_empty_content_blocks_raises_service_error():
    """Handles malformed or empty response payload safely."""
    mock_client = MagicMock()
    mock_client.converse.return_value = {"output": {"message": {"content": []}}}

    service = BedrockService(
        region="us-east-1",
        model_id="test-model",
        client=mock_client,
    )

    with pytest.raises(BedrockServiceError, match="empty content block"):
        service.generate_reply([Message(speaker="alex", text="Hello")])
