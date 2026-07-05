from fastapi import APIRouter, HTTPException
from rag.db import get_connection
import json

router = APIRouter(prefix="/api/working-sets", tags=["working_sets"])


@router.get("")
def list_working_sets() -> dict:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT id, name, source_ids, created_at, updated_at FROM working_sets ORDER BY name"
        ).fetchall()
    return {"working_sets": [_row_to_dict(r) for r in rows]}


@router.post("")
def create_working_set(payload: dict) -> dict:
    name = payload.get("name", "").strip()
    if not name:
        raise HTTPException(status_code=422, detail="name is required")
    source_ids = payload.get("source_ids", [])
    if not isinstance(source_ids, list):
        raise HTTPException(status_code=422, detail="source_ids must be a list")
    try:
        with get_connection() as conn:
            row = conn.execute(
                """INSERT INTO working_sets (name, source_ids)
                   VALUES (%s, %s::jsonb)
                   RETURNING id""",
                (name, json.dumps(source_ids)),
            ).fetchone()
            conn.commit()
        return {"id": str(row[0]), "name": name}
    except Exception as e:
        if "unique" in str(e).lower() or "duplicate" in str(e).lower():
            raise HTTPException(status_code=400, detail=f"working set name already exists: {name}")
        raise


@router.get("/{ws_id}")
def get_working_set(ws_id: str) -> dict:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT id, name, source_ids, created_at, updated_at FROM working_sets WHERE id = %s",
            (ws_id,),
        ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="working set not found")
    return _row_to_dict(row)


@router.patch("/{ws_id}")
def update_working_set(ws_id: str, payload: dict) -> dict:
    with get_connection() as conn:
        existing = conn.execute(
            "SELECT id, name, source_ids FROM working_sets WHERE id = %s",
            (ws_id,),
        ).fetchone()
    if existing is None:
        raise HTTPException(status_code=404, detail="working set not found")

    name = payload.get("name", "").strip()
    source_ids = payload.get("source_ids")

    if name:
        try:
            with get_connection() as conn:
                conn.execute(
                    "UPDATE working_sets SET name = %s, updated_at = now() WHERE id = %s",
                    (name, ws_id),
                )
                conn.commit()
        except Exception as e:
            if "unique" in str(e).lower() or "duplicate" in str(e).lower():
                raise HTTPException(status_code=400, detail=f"working set name already exists: {name}")
            raise

    if source_ids is not None and isinstance(source_ids, list):
        with get_connection() as conn:
            conn.execute(
                "UPDATE working_sets SET source_ids = %s::jsonb, updated_at = now() WHERE id = %s",
                (json.dumps(source_ids), ws_id),
            )
            conn.commit()

    return get_working_set(ws_id)


@router.delete("/{ws_id}")
def delete_working_set(ws_id: str) -> dict:
    with get_connection() as conn:
        result = conn.execute("DELETE FROM working_sets WHERE id = %s", (ws_id,))
        conn.commit()
    if result.rowcount == 0:
        raise HTTPException(status_code=404, detail="working set not found")
    return {"deleted": ws_id}


def _row_to_dict(row) -> dict:
    source_ids = row[2]
    if isinstance(source_ids, str):
        try:
            source_ids = json.loads(source_ids)
        except (json.JSONDecodeError, TypeError):
            source_ids = []
    source_ids = source_ids if isinstance(source_ids, list) else []
    return {
        "id": str(row[0]),
        "name": row[1],
        "source_ids": source_ids,
        "source_count": len(source_ids),
        "created_at": row[3].isoformat() if row[3] else None,
        "updated_at": row[4].isoformat() if row[4] else None,
    }
