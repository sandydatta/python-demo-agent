import io
from fastapi.testclient import TestClient
from PIL import Image
from pypdf import PdfReader, PdfWriter

from app.chroma_rag_service import get_chroma_rag_service
from app.fast_api_app import app
from app.pdf_parser import parse_pdf_bytes

client = TestClient(app)


def generate_sample_pdf_bytes() -> bytes:
    """Generates a sample PDF containing text and an embedded image in memory."""
    img = Image.new("RGB", (100, 100), color="blue")
    img_pdf_io = io.BytesIO()
    img.save(img_pdf_io, format="PDF")
    img_pdf_bytes = img_pdf_io.getvalue()

    writer = PdfWriter()
    img_reader = PdfReader(io.BytesIO(img_pdf_bytes))
    page = writer.add_blank_page(width=612, height=792)

    if len(img_reader.pages) > 0:
        page.merge_page(img_reader.pages[0])

    pdf_buffer = io.BytesIO()
    writer.write(pdf_buffer)
    return pdf_buffer.getvalue()


def test_chroma_rag_service_indexing_and_search():
    pdf_bytes = generate_sample_pdf_bytes()
    parsed = parse_pdf_bytes(pdf_bytes, filename="chroma_test.pdf")
    rag_service = get_chroma_rag_service()

    # Index PDF into ChromaDB
    index_res = rag_service.index_pdf(parsed)
    assert index_res["filename"] == "chroma_test.pdf"
    assert index_res["images_indexed"] >= 1

    # Test semantic search in ChromaDB
    search_res = rag_service.semantic_search("blue image query", top_k=2)
    assert isinstance(search_res, list)

    # Test Chroma RAG answer generation
    rag_res = rag_service.generate_rag_answer(query="What is in the document?", top_k=2)
    assert "query" in rag_res
    assert "answer" in rag_res
    assert isinstance(rag_res["retrieved_context"], list)


def test_chroma_pdf_api_endpoints():
    pdf_bytes = generate_sample_pdf_bytes()

    # 1. Test POST /api/v1/chroma/upload
    upload_res = client.post(
        "/api/v1/chroma/upload",
        files={"file": ("chroma_test.pdf", pdf_bytes, "application/pdf")},
    )
    assert upload_res.status_code == 201
    json_data = upload_res.json()
    assert json_data["filename"] == "chroma_test.pdf"
    assert "text_vector_ids" in json_data
    assert "image_vector_ids" in json_data

    # 2. Test POST /api/v1/chroma/query
    query_res = client.post(
        "/api/v1/chroma/query",
        json={"query": "Explain the contents of the document in ChromaDB", "top_k": 3},
    )
    assert query_res.status_code == 200
    query_json = query_res.json()
    assert "answer" in query_json
    assert query_json["query"] == "Explain the contents of the document in ChromaDB"

    # 3. Test GET /api/v1/chroma/stats
    stats_res = client.get("/api/v1/chroma/stats")
    assert stats_res.status_code == 200
    stats_json = stats_res.json()
    assert stats_json["vector_database"] == "ChromaDB"
    assert "indexed_vectors_count" in stats_json
