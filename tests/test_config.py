from rag.config import settings


def test_graph_config_defaults():
    assert settings.MEMGRAPH_URL == "bolt://localhost:7687"
    assert settings.MODEL_ENTITY_EXTRACTION == "qwen/qwen-2.5-7b-instruct"
    assert settings.MODEL_RELATIONSHIP_EXTRACTION == "qwen/qwen-2.5-7b-instruct"
    assert settings.RELATIONSHIP_CONFIDENCE_THRESHOLD == 0.75
    assert settings.ENTITY_DEDUP_COSINE_THRESHOLD == 0.92


def test_retrieval_config_defaults():
    assert settings.MODEL_RETRIEVAL_QUERY_VARIANTS
    assert settings.MODEL_RETRIEVAL_GRAPH
    assert settings.MODEL_RETRIEVAL_RERANKER
    assert settings.RETRIEVAL_RRF_K == 60
    assert settings.RETRIEVAL_SEED_COUNT == 10
    assert settings.RETRIEVAL_RESULT_COUNT == 5
    assert settings.RETRIEVAL_MAX_DECOMPOSED_QUERIES == 5
    assert settings.RETRIEVAL_FIRST_STAGE_TOP_N == 20
    assert settings.RETRIEVAL_FUSED_CANDIDATE_COUNT == 50
    assert settings.RETRIEVAL_ENTITY_SELECTION_COUNT == 5
    assert settings.RETRIEVAL_SECOND_HOP_SELECTION_COUNT == 5
    assert settings.RETRIEVAL_MAX_GRAPH_EXPANSION_MS_PER_SEED == 4000
    assert settings.RETRIEVAL_SAME_SOURCE_NEIGHBOR_WINDOW == 2
    assert settings.RETRIEVAL_SAME_SOURCE_NEIGHBOR_COUNT == 3
