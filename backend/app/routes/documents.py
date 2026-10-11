from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from app.document_ingestion import DocumentIngestionError, ingest_pdf


router = APIRouter()


class DocumentUploadResponse(BaseModel):
    document_id: int
    filename: str
    status: str
    chunk_count: int
    content_hash: str


@router.post("/documents", response_model=DocumentUploadResponse, status_code=201)
def upload_document(
    file: UploadFile = File(...),
    uploaded_by: int = Form(...),
) -> DocumentUploadResponse:
    try:
        result = ingest_pdf(
            content=file.file.read(),
            filename=file.filename or "upload.pdf",
            uploaded_by=uploaded_by,
        )
    except DocumentIngestionError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from None
    finally:
        file.file.close()

    return DocumentUploadResponse(
        document_id=result.document_id,
        filename=result.filename,
        status=result.status,
        chunk_count=result.chunk_count,
        content_hash=result.content_hash,
    )
