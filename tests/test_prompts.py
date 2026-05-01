from rag import prompts


def test_all_prompt_constants_are_non_empty_strings():
    constants = [
        prompts.DOCUMENT_PROFILING,
        prompts.CHUNK_VALIDATION,
        prompts.ENTITY_EXTRACTION,
        prompts.RELATIONSHIP_EXTRACTION,
        prompts.PROPOSITION_DECOMPOSITION,
        prompts.ANSWER_GENERATION,
        prompts.COMMUNITY_SUMMARIZATION,
        prompts.QUERY_VARIANTS,
        prompts.ENTITY_SELECTION,
        prompts.ENTITY_QUERY_GENERATION,
        prompts.SECOND_HOP_ENTITY_SELECTION,
        prompts.INSIGHT_EXTRACTION,
    ]
    for c in constants:
        assert isinstance(c, str) and c.strip(), f"Expected non-empty string, got {c!r}"


def test_insight_extraction_prompt_placeholders():
    from rag.prompts import INSIGHT_EXTRACTION
    assert "{chunk}" in INSIGHT_EXTRACTION
    assert "insights" in INSIGHT_EXTRACTION
    assert "topics" in INSIGHT_EXTRACTION.lower()
