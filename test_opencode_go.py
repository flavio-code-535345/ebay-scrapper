"""Tests for OpenCode Go assessor (HTTP OpenAI-compatible client)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from ai_providers.opencode_go import OpenCodeGoAssessor


@pytest.fixture
def assessor(monkeypatch):
    monkeypatch.setenv("OPENCODE_GO_API_KEY", "test-key")
    monkeypatch.setenv("OPENCODE_GO_MODEL", "grok-4.5")
    return OpenCodeGoAssessor()


def test_disabled_without_key(monkeypatch):
    monkeypatch.delenv("OPENCODE_GO_API_KEY", raising=False)
    monkeypatch.delenv("OPENCODE_API_KEY", raising=False)
    a = OpenCodeGoAssessor()
    assert a.enabled is False
    assert a.assess_deal({"title": "x", "price": 1}) is None


def test_chat_parses_content(assessor):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": (
                        '{"deal_rating":"Good","confidence_score":80,'
                        '"visual_findings":[],"red_flags":[],'
                        '"fair_market_estimate":"€50","itemized_resale_estimates":[],'
                        '"estimated_total_cost":20,"estimated_gross_profit":30,'
                        '"verdict_summary":"Solid flip","potential_scam":false,'
                        '"scam_warning":""}'
                    )
                }
            }
        ]
    }
    with patch.object(assessor._session, "post", return_value=mock_resp) as post:
        result = assessor.assess_deal(
            {
                "title": "Zelda Switch",
                "price": 20,
                "condition": "Used",
                "seller_rating": 99,
                "shipping": "5",
                "description": "Works",
                "image_urls": [],
            }
        )
    assert result is not None
    assert result["ai_deal_rating"] == "Good"
    assert result["ai_assessed"] is True
    assert post.called
    kwargs = post.call_args.kwargs
    assert kwargs["json"]["model"] == "grok-4.5"
    assert kwargs["headers"]["Authorization"] == "Bearer test-key"


def test_rate_limit_returns_marker(assessor):
    mock_resp = MagicMock()
    mock_resp.status_code = 429
    mock_resp.text = "rate limited"
    with patch.object(assessor._session, "post", return_value=mock_resp):
        results = assessor.assess_deals_batch([{"title": "Game Lot", "price": 10, "description": "", "image_urls": []}])
    assert len(results) == 1
    assert results[0]["ai_error_type"] == "rate_limit"
