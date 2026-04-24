import base64
from unittest.mock import MagicMock, patch

import pytest

from rag.image_description import describe_image


@patch("rag.image_description.requests.post")
def test_describe_image_returns_description(mock_post):
    mock_post.return_value.json.return_value = {
        "choices": [{"message": {"content": "A bar chart showing sales data."}}]
    }
    mock_post.return_value.raise_for_status = MagicMock()

    result = describe_image(b"\x89PNG\r\n", "image/png")

    assert result == "A bar chart showing sales data."


@patch("rag.image_description.requests.post")
def test_describe_image_sends_base64_encoded_image(mock_post):
    mock_post.return_value.json.return_value = {
        "choices": [{"message": {"content": "description"}}]
    }
    mock_post.return_value.raise_for_status = MagicMock()
    image_bytes = b"fake-image-bytes"

    describe_image(image_bytes, "image/jpeg")

    call_json = mock_post.call_args.kwargs["json"]
    content = call_json["messages"][0]["content"]
    image_block = next(b for b in content if b["type"] == "image_url")
    expected_b64 = base64.b64encode(image_bytes).decode()
    assert f"data:image/jpeg;base64,{expected_b64}" in image_block["image_url"]["url"]


@patch("rag.image_description.requests.post")
def test_describe_image_uses_configured_model(mock_post, monkeypatch):
    monkeypatch.setattr("rag.image_description.settings.MODEL_IMAGE_DESCRIPTION", "test/model")
    mock_post.return_value.json.return_value = {
        "choices": [{"message": {"content": "ok"}}]
    }
    mock_post.return_value.raise_for_status = MagicMock()

    describe_image(b"bytes", "image/png")

    assert mock_post.call_args.kwargs["json"]["model"] == "test/model"


@patch("rag.image_description.requests.post")
def test_describe_image_raises_on_http_error(mock_post):
    import requests as req
    mock_post.return_value.raise_for_status.side_effect = req.HTTPError("500 Server Error")

    with pytest.raises(req.HTTPError):
        describe_image(b"bytes", "image/png")
