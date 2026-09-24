import logging
from typing import Any, Dict, Optional
from fastapi import APIRouter, File, HTTPException, UploadFile, status
from pydantic import BaseModel, Field

from app.pdf_parser import parse_pdf_bytes
from app.pinecone_rag_service import get_pinecone_rag_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/pdf", tags=["PDF RAG"])


class QueryRequest(BaseModel):
    query: str = Field(..., description="Semantic search query string")
    top_k: int = Field(3, ge=1, le=20, description="Number of context vectors to retrieve")


class QueryResponse(BaseModel):
    query: str
    answer: str
    retrieved_context: list[Dict[str, Any]]
    prompt_used: str


class UploadResponse(BaseModel):
    filename: str
    total_pages: int
    text_chunks_indexed: int
    images_indexed: int
    text_vector_ids: list[str]
    image_vector_ids: list[str]


@router.post(
    "/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload PDF and embed text/images separately into Pinecone",
)
async def upload_pdf(file: UploadFile = File(...)) -> UploadResponse:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must be a valid PDF document (.pdf).",
        )

    try:
        content = await file.read()
        parsed_pdf = parse_pdf_bytes(content, filename=file.filename)
        rag_service = get_pinecone_rag_service()
        result = rag_service.index_pdf(parsed_pdf)
        return UploadResponse(**result)
    except Exception as err:
        logger.error("Error processing PDF upload '%s': %s", file.filename, err)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process and index PDF: {err!s}",
        ) from err


@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Semantic search Pinecone and generate RAG LLM answer",
)
def query_pdf_rag(request: QueryRequest) -> QueryResponse:
    try:
        rag_service = get_pinecone_rag_service()
        result = rag_service.generate_rag_answer(query=request.query, top_k=request.top_k)
        return QueryResponse(**result)
    except Exception as err:
        logger.error("Error generating RAG answer for query '%s': %s", request.query, err)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"RAG query execution failed: {err!s}",
        ) from err


@router.get(
    "/stats",
    summary="Get vector index statistics",
)
def get_vector_stats() -> Dict[str, Any]:
    rag_service = get_pinecone_rag_service()
    in_mem_count = rag_service.in_memory_store.count()
    return {
        "pinecone_index_name": rag_service.index_name,
        "pinecone_connected": rag_service.pinecone_index is not None,
        "indexed_vectors_count": in_mem_count,
    }
