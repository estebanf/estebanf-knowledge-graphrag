from fastapi import APIRouter, HTTPException

from rag.db import get_connection

router = APIRouter(prefix="/api/answers", tags=["answers"])


@router.get("")
def list_answers_api(
    limit: int = 20,
    offset: int = 0,
) -> dict:
    with get_connection() as conn:
        total = conn.execute("SELECT count(*) FROM saved_answers").fetchone()[0]
        rows = conn.execute(
            """SELECT id, question, answer, model, params, evidence_snapshot, created_at
               FROM saved_answers ORDER BY created_at DESC LIMIT %s OFFSET %s""",
            (limit, offset),
        ).fetchall()
    return {
        "answers": [_row_to_answer(r) for r in rows],
        "total": total,
        "limit": limit,
        "offset": offset,
    }


@router.post("")
def save_answer(payload: dict) -> dict:
    required = ["question", "answer"]
    for field in required:
        if field not in payload or not payload[field]:
            raise HTTPException(status_code=422, detail=f"missing required field: {field}")

    evidence = payload.get("evidence", [])
    if not isinstance(evidence, list):
        raise HTTPException(status_code=422, detail="evidence must be a list")
    for item in evidence:
        if not isinstance(item, dict):
            raise HTTPException(status_code=422, detail="each evidence item must be an object")
        for key in ("source_id", "source_name", "text"):
            val = item.get(key, "")
            if not isinstance(val, str):
                raise HTTPException(status_code=422, detail=f"evidence.{key} must be a string")
            if len(val) > 100_000:
                raise HTTPException(status_code=422, detail=f"evidence.{key} too long")

    import json
    with get_connection() as conn:
        row = conn.execute(
            """INSERT INTO saved_answers (question, answer, model, params, evidence_snapshot)
               VALUES (%s, %s, %s, %s::jsonb, %s::jsonb)
               RETURNING id""",
            (
                payload["question"],
                payload["answer"],
                payload.get("model", ""),
                json.dumps(payload.get("params", {})),
                json.dumps(evidence),
            ),
        ).fetchone()
    return {"id": str(row[0])}


@router.get("/{answer_id}")
def get_answer(answer_id: str) -> dict:
    with get_connection() as conn:
        row = conn.execute(
            """SELECT id, question, answer, model, params, evidence_snapshot, created_at
               FROM saved_answers WHERE id = %s""",
            (answer_id,),
        ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="answer not found")
    return _row_to_answer(row)


@router.delete("/{answer_id}")
def delete_answer(answer_id: str) -> dict:
    with get_connection() as conn:
        result = conn.execute(
            "DELETE FROM saved_answers WHERE id = %s", (answer_id,)
        )
    if result.rowcount == 0:
        raise HTTPException(status_code=404, detail="answer not found")
    return {"deleted": answer_id}


def _row_to_answer(row) -> dict:
    import json
    evidence = row[5]
    if isinstance(evidence, str):
        try:
            evidence = json.loads(evidence)
        except (json.JSONDecodeError, TypeError):
            evidence = []
    params = row[4]
    if isinstance(params, str):
        try:
            params = json.loads(params)
        except (json.JSONDecodeError, TypeError):
            params = {}
    return {
        "id": str(row[0]),
        "question": row[1],
        "answer": row[2],
        "model": row[3] or "",
        "params": params if isinstance(params, dict) else {},
        "evidence_snapshot": evidence if isinstance(evidence, list) else [],
        "created_at": row[6].isoformat() if row[6] else None,
    }
