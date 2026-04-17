from unittest.mock import patch, MagicMock
from rag.profiling import DocumentProfile, profile_document, _DEFAULT_PROFILE

SAMPLE_MARKDOWN = """# Introduction

This is a well-structured document with consistent headings.

## Section One

Some prose content here.

## Section Two

More content.
"""


def test_profile_document_returns_dataclass(monkeypatch):
    monkeypatch.setattr("rag.profiling.settings.OPENROUTER_API_KEY", "test-key")
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {
        "choices": [{
            "message": {
                "content": '{"structure_type": "well-structured", "heading_consistency": "consistent", "content_density": "uniform", "primary_content_type": "prose", "avg_section_length": "medium", "has_tables": false, "has_code_blocks": false, "domain": "general"}'
            }
        }]
    }
    with patch("rag.profiling.requests.post", return_value=mock_resp):
        result = profile_document(SAMPLE_MARKDOWN)
    assert isinstance(result, DocumentProfile)
    assert result.structure_type == "well-structured"
    assert result.domain == "general"
    assert result.has_tables is False


def test_profile_document_returns_default_on_error(monkeypatch):
    monkeypatch.setattr("rag.profiling.settings.OPENROUTER_API_KEY", "test-key")
    with patch("rag.profiling.requests.post", side_effect=Exception("network error")):
        result = profile_document(SAMPLE_MARKDOWN)
    assert result == _DEFAULT_PROFILE


def test_profile_document_no_api_key(monkeypatch):
    monkeypatch.setattr("rag.profiling.settings.OPENROUTER_API_KEY", "")
    result = profile_document(SAMPLE_MARKDOWN)
    assert result == _DEFAULT_PROFILE
