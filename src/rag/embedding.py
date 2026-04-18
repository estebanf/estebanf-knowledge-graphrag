import requests

from rag.config import settings

_BATCH_SIZE = 32


def get_embeddings(texts: list[str]) -> list[list[float]]:
    """Call OpenRouter embeddings API in batches. Returns list of embedding vectors."""
    if not settings.OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY is required for embedding generation")

    results: list[list[float]] = []
    for i in range(0, len(texts), _BATCH_SIZE):
        batch = texts[i : i + _BATCH_SIZE]
        resp = requests.post(
            "https://openrouter.ai/api/v1/embeddings",
            headers={
                "Authorization": f"Bearer {settings.OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
            json={"model": settings.MODEL_EMBEDDING, "input": batch},
            timeout=120,
        )
        resp.raise_for_status()
        body = resp.json()
        if "data" not in body:
            raise RuntimeError(f"Embedding API error: {body}")
        data = body["data"]
        batch_vectors = [item["embedding"] for item in sorted(data, key=lambda x: x["index"])]
        results.extend(batch_vectors)
    return results


def embed_and_store_chunks(conn, chunk_rows: list[tuple[str, str]]) -> None:
    """
    Generate embeddings for chunks and update the DB.
    chunk_rows: list of (chunk_id, content) tuples
    """
    if not chunk_rows:
        return

    ids = [row[0] for row in chunk_rows]
    texts = [row[1] for row in chunk_rows]
    vectors = get_embeddings(texts)

    with conn.cursor() as cur:
        cur.executemany(
            "UPDATE chunks SET embedding = %s::vector WHERE id = %s",
            [(f"[{','.join(str(v) for v in vec)}]", chunk_id)
             for chunk_id, vec in zip(ids, vectors)],
        )
