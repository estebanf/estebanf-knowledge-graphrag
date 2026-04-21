from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from rag.answer_models import get_supported_answer_models
from rag.answering import stream_answer
from rag.api.schemas import AnswerRequest


router = APIRouter(prefix="/api/answer", tags=["answer"])


@router.get("/models")
def answer_models() -> dict[str, object]:
    return {"models": get_supported_answer_models()}


@router.post("/stream")
def answer_stream(payload: AnswerRequest) -> StreamingResponse:
    return StreamingResponse(
        stream_answer(
            query=payload.query,
            model=payload.model,
            source_ids=payload.source_ids,
            filters=payload.filters,
            seed_count=payload.seed_count,
            result_count=payload.result_count,
            rrf_k=payload.rrf_k,
            entity_confidence_threshold=payload.entity_confidence_threshold,
            first_hop_similarity_threshold=payload.first_hop_similarity_threshold,
            second_hop_similarity_threshold=payload.second_hop_similarity_threshold,
        ),
        media_type="text/event-stream",
    )
