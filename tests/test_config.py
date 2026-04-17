from rag.config import settings


def test_graph_config_defaults():
    assert settings.MEMGRAPH_URL == "bolt://localhost:7687"
    assert settings.MODEL_ENTITY_EXTRACTION == "qwen/qwen2.5-7b-instruct"
    assert settings.MODEL_RELATIONSHIP_EXTRACTION == "qwen/qwen2.5-7b-instruct"
    assert settings.RELATIONSHIP_CONFIDENCE_THRESHOLD == 0.75
    assert settings.ENTITY_DEDUP_COSINE_THRESHOLD == 0.92
