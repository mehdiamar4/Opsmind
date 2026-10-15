from functools import partial
from typing import Optional
from uuid import uuid4

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.exc import SQLAlchemyError

from app.rag import RAGConfigurationError, answer_question
from app.usage import UsageRecordingError, record_usage_event


router = APIRouter()


class RAGChatRequest(BaseModel):
    question: str = Field(min_length=1)
    document_id: Optional[int] = Field(default=None, gt=0)


class RAGSource(BaseModel):
    document_id: int
    filename: str
    page_number: int
    section_title: Optional[str]
    chunk_index: int
    distance: float


class RAGChatResponse(BaseModel):
    answer: str
    sources: list[RAGSource]


@router.post("/rag/chat", response_model=RAGChatResponse)
def rag_chat(request: RAGChatRequest) -> RAGChatResponse:
    request_id = uuid4()
    try:
        result = answer_question(
            request.question,
            document_id=request.document_id,
            usage_recorder=partial(record_usage_event, request_id=request_id),
        )
    except RAGConfigurationError:
        raise HTTPException(status_code=503, detail="RAG service is not configured.") from None
    except UsageRecordingError:
        raise HTTPException(status_code=503, detail="Usage metrics could not be recorded.") from None
    except SQLAlchemyError:
        raise HTTPException(status_code=503, detail="Document retrieval is unavailable.") from None
    except Exception:
        raise HTTPException(status_code=502, detail="RAG provider request failed.") from None

    return RAGChatResponse(
        answer=result.answer,
        sources=[
            RAGSource(
                document_id=chunk.document_id,
                filename=chunk.filename,
                page_number=chunk.page_number,
                section_title=chunk.section_title,
                chunk_index=chunk.chunk_index,
                distance=chunk.distance,
            )
            for chunk in result.sources
        ],
    )
