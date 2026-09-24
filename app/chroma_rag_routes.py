import logging
from typing import Any, Dict
from fastapi import APIRouter, File, HTTPException, UploadFile, status
from pydantic import BaseModel, Field

from app.chroma_rag_service import get_chroma_rag_service
from app.pdf_parser import parse_pdf_bytes

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/chroma", tags=["ChromaDB PDF RAG"])


class ChromaQueryRequest(BaseModel):
    query: str = Field(..., description="Semantic search query string")
    top_k: int = Field(3, ge=1, le=20, description="Number of context vectors to retrieve")


class ChromaQueryResponse(BaseModel):
    query: str
    answer: str
    retrieved_context: list[Dict[str, Any]]
    prompt_used: str


class ChromaUploadResponse(BaseModel):
    filename: str
    total_pages: int
    text_chunks_indexed: int
    images_indexed: int
    text_vector_ids: list[str]
    image_vector_ids: list[str]


@router.post(
    "/upload",
    response_model=ChromaUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload PDF and embed text/images separately into ChromaDB",
)
async def upload_pdf_chroma(file: UploadFile = File(...)) -> ChromaUploadResponse:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must be a valid PDF document (.pdf).",
        )

    try:
        content = await file.read()
        parsed_pdf = parse_pdf_bytes(content, filename=file.filename)
        service = get_chroma_rag_service()
        result = service.index_pdf(parsed_pdf)
        return ChromaUploadResponse(**result)
    except Exception as err:
        logger.error("Error processing Chroma PDF upload '%s': %s", file.filename, err)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process and index PDF into ChromaDB: {err!s}",
        ) from err


@router.post(
    "/query",
    response_model=ChromaQueryResponse,
    summary="Semantic search ChromaDB and generate RAG LLM answer",
)
def query_chroma_rag(request: ChromaQueryRequest) -> ChromaQueryResponse:
    try:
        service = get_chroma_rag_service()
        result = service.generate_rag_answer(query=request.query, top_k=request.top_k)
        return ChromaQueryResponse(**result)
    except Exception as err:
        logger.error("Error generating Chroma RAG answer for query '%s': %s", request.query, err)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Chroma RAG query execution failed: {err!s}",
        ) from err


@router.get(
    "/stats",
    summary="Get ChromaDB collection statistics",
)
def get_chroma_stats() -> Dict[str, Any]:
    service = get_chroma_rag_service()
    return {
        "collection_name": service.collection_name,
        "indexed_vectors_count": service.collection.count(),
        "vector_database": "ChromaDB",
    }
