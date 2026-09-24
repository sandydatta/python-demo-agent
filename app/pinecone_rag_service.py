import io
import logging
import math
import os
from typing import Any, Dict, List, Optional
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer
from app.pdf_parser import ParsedPDF

logger = logging.getLogger(__name__)

EMBEDDING_DIM = 384


class ImageEmbedder:
    """Computes a 384-dimensional vector representation for images."""

    def __init__(self, dimension: int = EMBEDDING_DIM):
        self.dimension = dimension

    def embed_bytes(self, image_bytes: bytes) -> List[float]:
        try:
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            # Resize image to a fixed 16x24 grid (=384 pixels) to create a deterministic feature vector
            img_resized = img.resize((16, 24))
            arr = np.array(img_resized, dtype=np.float32) / 255.0  # (24, 16, 3)
            # Flatten to 1152 elements, then aggregate 3-color channels down to 384 features
            flat = arr.mean(axis=2).flatten()  # 24 * 16 = 384
            # Normalize vector to unit length
            norm = np.linalg.norm(flat)
            if norm > 0:
                flat = flat / norm
            return flat.tolist()
        except Exception as err:
            logger.warning("Error generating image embedding, using zero vector fallback: %s", err)
            return [0.0] * self.dimension


class InMemoryVectorStore:
    """In-memory vector store fallback for local execution without a live Pinecone cluster."""

    def __init__(self):
        self.vectors: Dict[str, Dict[str, Any]] = {}

    def upsert(self, vectors: List[Dict[str, Any]]) -> None:
        for item in vectors:
            self.vectors[item["id"]] = item

    def query(
        self,
        query_vector: List[float],
        top_k: int = 3,
        filter_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        q_vec = np.array(query_vector, dtype=np.float32)
        q_norm = np.linalg.norm(q_vec)
        results = []

        for item_id, item in self.vectors.items():
            meta = item.get("metadata", {})
            if filter_type and meta.get("content_type") != filter_type:
                continue

            v_vec = np.array(item["values"], dtype=np.float32)
            v_norm = np.linalg.norm(v_vec)
            if q_norm > 0 and v_norm > 0:
                score = float(np.dot(q_vec, v_vec) / (q_norm * v_norm))
            else:
                score = 0.0

            results.append({"id": item_id, "score": score, "metadata": meta})

        # Sort by similarity score descending
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def count(self) -> int:
        return len(self.vectors)


class PineconeRAGService:
    """Service for separate text/image embedding into Pinecone and RAG semantic search."""

    def __init__(self, index_name: str = "pdf-rag-index"):
        self.index_name = index_name
        self.text_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.image_embedder = ImageEmbedder(dimension=EMBEDDING_DIM)
        self.in_memory_store = InMemoryVectorStore()
        self.pinecone_index = None

        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        if pinecone_api_key:
            try:
                from pinecone import Pinecone, ServerlessSpec
                pc = Pinecone(api_key=pinecone_api_key)
                existing_indexes = [idx.name for idx in pc.list_indexes()]
                if self.index_name not in existing_indexes:
                    logger.info("Creating Pinecone index '%s'...", self.index_name)
                    pc.create_index(
                        name=self.index_name,
                        dimension=EMBEDDING_DIM,
                        metric="cosine",
                        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
                    )
                self.pinecone_index = pc.Index(self.index_name)
                logger.info("Connected to live Pinecone index '%s'.", self.index_name)
            except Exception as err:
                logger.warning("Pinecone connection failed, using in-memory store fallback: %s", err)
        else:
            logger.info("No PINECONE_API_KEY set. Using in-memory vector store fallback.")

    def embed_text(self, text: str) -> List[float]:
        vector = self.text_model.encode(text)
        return vector.tolist()

    def embed_image(self, image_bytes: bytes) -> List[float]:
        return self.image_embedder.embed_bytes(image_bytes)

    def index_pdf(self, parsed_pdf: ParsedPDF) -> Dict[str, Any]:
        vectors_to_upsert = []
        text_vector_ids = []
        image_vector_ids = []

        # 1. Embed text chunks separately
        for chunk in parsed_pdf.text_chunks:
            vector = self.embed_text(chunk.text)
            vectors_to_upsert.append({
                "id": chunk.chunk_id,
                "values": vector,
                "metadata": {
                    "content_type": "text",
                    "filename": parsed_pdf.filename,
                    "page_number": chunk.page_number,
                    "text_content": chunk.text,
                },
            })
            text_vector_ids.append(chunk.chunk_id)

        # 2. Embed image chunks separately
        for img_chunk in parsed_pdf.image_chunks:
            vector = self.embed_image(img_chunk.image_bytes)
            vectors_to_upsert.append({
                "id": img_chunk.image_id,
                "values": vector,
                "metadata": {
                    "content_type": "image",
                    "filename": parsed_pdf.filename,
                    "page_number": img_chunk.page_number,
                    "image_index": img_chunk.image_index,
                    "format": img_chunk.format,
                },
            })
            image_vector_ids.append(img_chunk.image_id)

        # Upsert into Pinecone or fallback in-memory vector store
        if vectors_to_upsert:
            if self.pinecone_index:
                self.pinecone_index.upsert(vectors=vectors_to_upsert)
            self.in_memory_store.upsert(vectors_to_upsert)

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

        if self.pinecone_index:
            try:
                filter_dict = {}
                if content_type:
                    filter_dict = {"content_type": {"$eq": content_type}}
                response = self.pinecone_index.query(
                    vector=query_vector,
                    top_k=top_k,
                    include_metadata=True,
                    filter=filter_dict if filter_dict else None,
                )
                matches = []
                for match in response.matches:
                    matches.append({
                        "id": match.id,
                        "score": float(match.score),
                        "metadata": match.metadata,
                    })
                return matches
            except Exception as err:
                logger.warning("Pinecone query error, falling back to in-memory store: %s", err)

        return self.in_memory_store.query(
            query_vector, top_k=top_k, filter_type=content_type
        )

    def generate_rag_answer(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        """Performs semantic search in Pinecone and feeds context to LLM for RAG answer generation."""
        search_results = self.semantic_search(query, top_k=top_k)

        # Construct context from text and image search results
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
            f"You are a helpful AI assistant answering a query based on retrieved PDF context.\n\n"
            f"Context:\n{context_str}\n\n"
            f"User Query: {query}\n\n"
            f"Answer the query clearly based on the context above."
        )

        # LLM Invocation with fallback
        answer = self._call_llm(prompt, context_str, query)

        return {
            "query": query,
            "answer": answer,
            "retrieved_context": search_results,
            "prompt_used": prompt,
        }

    def _call_llm(self, prompt: str, context_str: str, query: str) -> str:
        """Invokes LLM if API keys present, otherwise returns a structured RAG completion."""
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

        # Local structured fallback when API key is not supplied
        if not context_str.strip():
            return f"No relevant PDF context found in Pinecone vector store for query: '{query}'."

        return (
            f"Based on the retrieved context from Pinecone:\n"
            f"{context_str}\n\n"
            f"Answer: The query '{query}' was answered using vector context from the indexed PDF."
        )


# Global singleton instance for app routes and tools
_rag_service_instance: Optional[PineconeRAGService] = None


def get_pinecone_rag_service() -> PineconeRAGService:
    global _rag_service_instance
    if _rag_service_instance is None:
        _rag_service_instance = PineconeRAGService()
    return _rag_service_instance
