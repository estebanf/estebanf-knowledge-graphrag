from rag.answering import _build_answer_prompt


def test_build_answer_prompt_emphasizes_relevance_and_gaps():
    prompt = _build_answer_prompt(
        "What does the retrieval say about cost structure?",
        {"retrieval_results": [{"chunk_id": "chunk-1", "chunk": "Evidence"}]},
    )

    assert 'User question: "What does the retrieval say about cost structure?"' in prompt
    assert "Use only the retrieved results below." in prompt
    assert (
        "Write a comprehensive narrative answer that clearly answers the question first, "
        "then elaborates on the supporting evidence." in prompt
    )
    assert "Begin with a direct thesis that answers the exact question." in prompt
    assert "Write 3 to 5 cohesive paragraphs, not bullets." in prompt
    assert "Build the answer around the most relevant evidence, not around all major themes" in prompt
    assert "Prioritize evidence that matches the narrow angle of the question." in prompt
    assert "Use broader AI-native themes only when they strengthen the insurance positioning." in prompt
    assert "Do not elevate a concept into the main answer just because it appears in a high-scoring chunk." in prompt
    assert "End with a brief statement of what the retrieval does not establish" in prompt
    assert "Do not mention chunk IDs, scores, or retrieval mechanics." in prompt
    assert "Prefer a strong narrative arc: answer, reasoning, implications, limitations." in prompt
    assert '"chunk_id": "chunk-1"' in prompt
