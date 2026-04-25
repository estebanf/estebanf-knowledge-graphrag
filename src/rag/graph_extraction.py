import json
import re
import uuid

import requests

from rag import prompts
from rag.config import settings
from rag.embedding import get_embeddings

_ENTITY_TYPES = ["ORGANIZATION", "PERSON", "POLICY", "PRODUCT", "REGULATION", "CONCEPT", "LOCATION"]


def extract_entities(chunk_content: str) -> list[dict]:
    if not settings.OPENROUTER_API_KEY:
        return []
    try:
        resp = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {settings.OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": settings.MODEL_ENTITY_EXTRACTION,
                "messages": [{"role": "user", "content": prompts.ENTITY_EXTRACTION.format(
                    types=", ".join(_ENTITY_TYPES),
                    text=chunk_content[:4000],
                )}],
                "temperature": 0,
            },
            timeout=60,
        )
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"]
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            parsed = next((v for v in parsed.values() if isinstance(v, list)), [])
        return [e for e in parsed if isinstance(e, dict)]
    except Exception:
        return []


def extract_relationships(chunk_content: str, entities: list[dict]) -> list[dict]:
    if not entities or not settings.OPENROUTER_API_KEY:
        return []
    try:
        entity_names = [e["canonical_name"] for e in entities]
        resp = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {settings.OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": settings.MODEL_RELATIONSHIP_EXTRACTION,
                "messages": [{"role": "user", "content": prompts.RELATIONSHIP_EXTRACTION.format(
                    entity_names=", ".join(entity_names),
                    text=chunk_content[:4000],
                )}],
                "temperature": 0,
            },
            timeout=60,
        )
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"]
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
        rels = json.loads(raw)
        return [r for r in rels if r.get("confidence", 0) >= settings.RELATIONSHIP_CONFIDENCE_THRESHOLD]
    except Exception:
        return []


def store_entities_and_edges(
    conn,
    driver,
    chunk_id: str,
    source_id: str,
    entities: list[dict],
    relationships: list[dict],
) -> list[str]:
    entity_ids: list[str] = []
    name_to_id: dict[str, str] = {}

    if not entities:
        return entity_ids

    names = [e["canonical_name"] for e in entities]
    try:
        vecs = get_embeddings(names)
    except Exception:
        vecs = [None] * len(entities)

    if len(vecs) != len(names):
        vecs = [None] * len(entities)

    with driver.session() as session:
        for entity, vec in zip(entities, vecs):
            entity_id = str(uuid.uuid4())
            entity_ids.append(entity_id)
            name_to_id[entity["canonical_name"]] = entity_id

            embedding_str = f"[{','.join(str(v) for v in vec)}]" if vec is not None else None

            conn.execute(
                """INSERT INTO entities (id, canonical_name, entity_type, aliases, source_id, embedding)
                   VALUES (%s, %s, %s, %s, %s, %s::vector)
                   ON CONFLICT DO NOTHING""",
                (
                    entity_id,
                    entity["canonical_name"],
                    entity["entity_type"],
                    entity.get("aliases", []),
                    source_id,
                    embedding_str,
                ),
            )

            session.run(
                "MERGE (e:Entity {entity_id: $entity_id}) "
                "SET e.canonical_name = $canonical_name, e.entity_type = $entity_type",
                entity_id=entity_id,
                canonical_name=entity["canonical_name"],
                entity_type=entity["entity_type"],
            )
            session.run(
                "MATCH (c:Chunk {chunk_id: $chunk_id}), (e:Entity {entity_id: $entity_id}) "
                "MERGE (c)-[:MENTIONS {confidence: $confidence}]->(e)",
                chunk_id=chunk_id,
                entity_id=entity_id,
                confidence=1.0,
            )

        for rel in relationships:
            e1_id = name_to_id.get(rel.get("source", ""))
            e2_id = name_to_id.get(rel.get("target", ""))
            if not e1_id or not e2_id:
                continue
            session.run(
                "MATCH (e1:Entity {entity_id: $e1_id}), (e2:Entity {entity_id: $e2_id}) "
                "MERGE (e1)-[:RELATED_TO {type: $type, confidence: $confidence, chunk_id: $chunk_id}]->(e2)",
                e1_id=e1_id,
                e2_id=e2_id,
                type=rel["type"],
                confidence=rel["confidence"],
                chunk_id=chunk_id,
            )

    return entity_ids


def extract_and_store_graph(
    conn,
    driver,
    source_id: str,
    job_id: str,
    chunk_rows: list[tuple[str, str]],
) -> None:
    for chunk_id, content in chunk_rows:
        entities = extract_entities(content)
        store_entities_and_edges(conn, driver, chunk_id, source_id, entities, [])
    conn.commit()
