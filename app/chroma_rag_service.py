import logging
import os
from typing import Any, Dict, List, Optional

import chromadb
from sentence_transformers import SentenceTransformer

from app.pdf_parser import ParsedPDF
from app.pinecone_rag_service import EMBEDDING_DIM, ImageEmbedder

logger = logging.getLogger(__name__)


class ChromaRAGService:
    """Service for separate text/image embedding into ChromaDB and RAG semantic search."""

    def __init__(self, collection_name: str = "chroma_pdf_rag"):
        self.collection_name = collection_name
        self.text_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.image_embedder = ImageEmbedder(dimension=EMBEDDING_DIM)

        # Initialize ephemeral or persistent ChromaDB client
        self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection(name=self.collection_name)
        logger.info("ChromaRAGService initialized with collection '%s'.", self.collection_name)

    def embed_text(self, text: str) -> List[float]:
        vector = self.text_model.encode(text)
        return vector.tolist()

    def embed_image(self, image_bytes: bytes) -> List[float]:
        return self.image_embedder.embed_bytes(image_bytes)

    def index_pdf(self, parsed_pdf: ParsedPDF) -> Dict[str, Any]:
        ids = []
        embeddings = []
        metadatas = []
        documents = []

        text_vector_ids = []
        image_vector_ids = []

        # 1. Embed text chunks separately
        for chunk in parsed_pdf.text_chunks:
            vec = self.embed_text(chunk.text)
            ids.append(chunk.chunk_id)
            embeddings.append(vec)
            metadatas.append({
                "content_type": "text",
                "filename": parsed_pdf.filename,
                "page_number": chunk.page_number,
                "text_content": chunk.text,
            })
            documents.append(chunk.text)
            text_vector_ids.append(chunk.chunk_id)

        # 2. Embed image chunks separately
        for img_chunk in parsed_pdf.image_chunks:
            vec = self.embed_image(img_chunk.image_bytes)
            ids.append(img_chunk.image_id)
            embeddings.append(vec)
            metadatas.append({
                "content_type": "image",
                "filename": parsed_pdf.filename,
                "page_number": img_chunk.page_number,
                "image_index": img_chunk.image_index,
                "format": img_chunk.format,
            })
            documents.append(f"[Image from {parsed_pdf.filename} page {img_chunk.page_number}]")
            image_vector_ids.append(img_chunk.image_id)

        # Upsert into ChromaDB collection
        if ids:
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents,
            )

        return {
            "filename": parsed_pdf.filename,
            "total_pages": parsed_pdf.num_pages,
            "text_chunks_indexed": len(text_vector_ids),
            "images_indexed": len(image_vector_ids),
            "text_vector_ids": text_vector_ids,
            "image_vector_ids": image_vector_ids,
        }

    def semantic_search(
        self, query: str, top_k: int = 3, content_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        query_vector = self.embed_text(query)

        where_filter = None
        if content_type:
            where_filter = {"content_type": content_type}

        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=min(top_k, max(1, self.collection.count())),
            where=where_filter,
            include=["embeddings", "metadatas", "documents", "distances"],
        )

        matches = []
        if results and "ids" in results and results["ids"]:
            item_ids = results["ids"][0]
            metadatas = results.get("metadatas", [[]])[0]
            distances = results.get("distances", [[]])[0]

            for i in range(len(item_ids)):
                # Chroma distance: cosine distance = 1 - cosine_similarity
                dist = float(distances[i]) if i < len(distances) else 0.0
                score = round(1.0 - dist, 4) if dist <= 1.0 else round(1.0 / (1.0 + dist), 4)

                matches.append({
                    "id": item_ids[i],
                    "score": score,
                    "metadata": metadatas[i] if i < len(metadatas) else {},
                })

        return matches

    def generate_rag_answer(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        """Performs semantic search in ChromaDB and feeds context to LLM for RAG answer generation."""
        if self.collection.count() == 0:
            return {
                "query": query,
                "answer": f"No relevant PDF context found in ChromaDB for query: '{query}'.",
                "retrieved_context": [],
                "prompt_used": "",
            }

        search_results = self.semantic_search(query, top_k=top_k)

        context_lines = []
        for idx, res in enumerate(search_results, start=1):
            meta = res.get("metadata", {})
            c_type = meta.get("content_type", "unknown")
            page = meta.get("page_number", "?")
            fname = meta.get("filename", "doc")
            if c_type == "text":
                text = meta.get("text_content", "")
                context_lines.append(f"[{idx}] (Text from {fname} page {page}): {text}")
            elif c_type == "image":
                img_fmt = meta.get("format", "image")
                context_lines.append(f"[{idx}] (Image from {fname} page {page}, format {img_fmt})")

        context_str = "\n".join(context_lines)

        prompt = (
            f"You are a helpful AI assistant answering a query based on retrieved ChromaDB PDF context.\n\n"
            f"Context:\n{context_str}\n\n"
            f"User Query: {query}\n\n"
            f"Answer the query clearly based on the context above."
        )

        answer = self._call_llm(prompt, context_str, query)

        return {
            "query": query,
            "answer": answer,
            "retrieved_context": search_results,
            "prompt_used": prompt,
        }

    def _call_llm(self, prompt: str, context_str: str, query: str) -> str:
        gemini_api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if gemini_api_key:
            try:
                from google import genai
                client = genai.Client(api_key=gemini_api_key)
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt,
                )
                return response.text
            except Exception as err:
                logger.warning("Gemini API call failed: %s", err)

        return (
            f"Based on retrieved context from ChromaDB:\n"
            f"{context_str}\n\n"
            f"Answer: The query '{query}' was answered using vector context from ChromaDB."
        )


_chroma_rag_service_instance: Optional[ChromaRAGService] = None


def get_chroma_rag_service() -> ChromaRAGService:
    global _chroma_rag_service_instance
    if _chroma_rag_service_instance is None:
        _chroma_rag_service_instance = ChromaRAGService()
    return _chroma_rag_service_instance
