"""Tests for PageIndex response handling without external API calls."""

import pytest

from rag_shootout.pageindex_pipeline import PageIndexError, PageIndexPipeline


class FakePageIndexClient:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def chat_completions(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


def _pipeline_with_response(response):
    pipeline = PageIndexPipeline.__new__(PageIndexPipeline)
    pipeline._client = FakePageIndexClient(response)
    pipeline._doc_id = "doc_test"
    return pipeline


def test_answer_extracts_openai_style_dict_response():
    pipeline = _pipeline_with_response(
        {"choices": [{"message": {"content": "A relevant PageIndex answer."}}]}
    )

    result = pipeline.answer("What happened?")

    assert result["answer"] == "A relevant PageIndex answer."
    assert result["doc_id"] == "doc_test"
    assert "raw_response" in result
    assert pipeline._client.calls[0]["doc_id"] == "doc_test"
    assert pipeline._client.calls[0]["messages"][0]["content"] == "What happened?"


def test_answer_raises_for_placeholder_failure_response():
    pipeline = _pipeline_with_response(
        {"choices": [{"message": {"content": "SKIPPED - no credits"}}]}
    )

    with pytest.raises(PageIndexError, match="failure/placeholder"):
        pipeline.answer("What happened?")


def test_answer_raises_when_response_shape_is_unknown():
    pipeline = _pipeline_with_response({"unexpected": "shape"})

    with pytest.raises(PageIndexError, match="Could not extract"):
        pipeline.answer("What happened?")
