import json
import re
import logging

import httpx

from rag.config import settings
from rag import prompts
from rag.embedding import get_embeddings

log = logging.getLogger(__name__)

_OPENCODE_URL = "https://opencode.ai/zen/go/v1/chat/completions"
_MODEL = "deepseek-v4-flash"
_MAX_CHUNK_CHARS = 4000


def _strip_fences(text: str) -> str:
    return re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())


def extract_insights_from_chunk(content: str) -> list[dict]:
    if not settings.OPENCODE_API_KEY:
        return []
    try:
        prompt = prompts.INSIGHT_EXTRACTION.format(chunk=content[:_MAX_CHUNK_CHARS])
        resp = httpx.post(
            _OPENCODE_URL,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {settings.OPENCODE_API_KEY}",
            },
            json={"model": _MODEL, "messages": [{"role": "user", "content": prompt}]},
            timeout=(10, 120),
        )
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"]
        return json.loads(_strip_fences(raw)).get("insights", [])
    except Exception as exc:
        log.warning("insight_extraction_failed", extra={"error": str(exc)})
        return []


def upsert_insight(conn, content: str, embedding: list[float]) -> tuple[str, bool]:
    emb_str = "[" + ",".join(str(v) for v in embedding) + "]"
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, 1 - (embedding <=> %s::vector) AS sim
            FROM insights
            ORDER BY embedding <=> %s::vector
            LIMIT 1
            """,
            (emb_str, emb_str),
        )
        row = cur.fetchone()
        if row and row[1] >= settings.INSIGHT_DEDUP_COSINE_THRESHOLD:
            return str(row[0]), False
        cur.execute(
            "INSERT INTO insights (content, embedding) VALUES (%s, %s::vector) RETURNING id",
            (content, emb_str),
        )
        new_id = cur.fetchone()[0]
    return str(new_id), True


def link_chunk_insight(conn, chunk_id: str, insight_id: str, topics: list[str]) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO chunk_insights (chunk_id, insight_id, topics)
            VALUES (%s, %s, %s)
            ON CONFLICT (chunk_id, insight_id) DO UPDATE SET topics = EXCLUDED.topics
            """,
            (chunk_id, insight_id, topics),
        )


def store_insight_in_graph(
    driver, chunk_id: str, insight_id: str, content: str, topics: list[str]
) -> None:
    with driver.session() as session:
        session.run(
            "MERGE (i:Insight {insight_id: $insight_id}) SET i.content = $content",
            insight_id=insight_id,
            content=content,
        )
        session.run(
            """
            MATCH (c:Chunk {chunk_id: $chunk_id}), (i:Insight {insight_id: $insight_id})
            MERGE (c)-[:CONTAINS {topics: $topics}]->(i)
            """,
            chunk_id=chunk_id,
            insight_id=insight_id,
            topics=topics,
        )


def link_related_insights(
    conn, driver, insight_id: str, embedding: list[float]
) -> None:
    k = settings.INSIGHT_LINK_TOP_K
    emb_str = "[" + ",".join(str(v) for v in embedding) + "]"

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, 1 - (embedding <=> %s::vector) AS sim, embedding
            FROM insights
            WHERE id != %s
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            (emb_str, insight_id, emb_str, k),
        )
        a_neighbors = cur.fetchall()

        for b_id, sim, b_emb_raw in a_neighbors:
            b_emb_str = "[" + ",".join(str(v) for v in b_emb_raw) + "]"
            cur.execute(
                """
                SELECT id FROM insights
                WHERE id != %s
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (b_id, b_emb_str, k),
            )
            b_neighbor_ids = {str(r[0]) for r in cur.fetchall()}

            if insight_id in b_neighbor_ids:
                with driver.session() as session:
                    session.run(
                        """
                        MATCH (a:Insight {insight_id: $a_id}), (b:Insight {insight_id: $b_id})
                        MERGE (a)-[:RELATED_TO {similarity: $sim}]->(b)
                        MERGE (b)-[:RELATED_TO {similarity: $sim}]->(a)
                        """,
                        a_id=insight_id,
                        b_id=str(b_id),
                        sim=float(sim),
                    )


def extract_and_store_insights(
    conn,
    driver,
    source_id: str,
    chunk_rows: list[tuple[str, str]],
) -> dict:
    chunks_processed = 0
    insights_extracted = 0
    insights_reused = 0

    for chunk_id, content in chunk_rows:
        raw_insights = extract_insights_from_chunk(content)
        if not raw_insights:
            chunks_processed += 1
            conn.commit()
            continue

        texts = [r["insight"] for r in raw_insights]
        embeddings = get_embeddings(texts)

        for raw, emb in zip(raw_insights, embeddings):
            insight_id, is_new = upsert_insight(conn, raw["insight"], emb)
            topics = raw.get("topics", [])
            link_chunk_insight(conn, chunk_id, insight_id, topics)
            store_insight_in_graph(driver, chunk_id, insight_id, raw["insight"], topics)
            if is_new:
                link_related_insights(conn, driver, insight_id, emb)
                insights_extracted += 1
            else:
                insights_reused += 1

        chunks_processed += 1
        conn.commit()

    return {
        "chunks_processed": chunks_processed,
        "insights_extracted": insights_extracted,
        "insights_reused": insights_reused,
    }
