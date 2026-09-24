import io
import logging
from dataclasses import dataclass, field
from typing import List, Optional
from PIL import Image
from pypdf import PdfReader

logger = logging.getLogger(__name__)


@dataclass
class PDFTextChunk:
    page_number: int
    text: str
    chunk_id: str


@dataclass
class PDFImageChunk:
    page_number: int
    image_index: int
    image_bytes: bytes
    format: str
    image_id: str


@dataclass
class ParsedPDF:
    filename: str
    num_pages: int
    text_chunks: List[PDFTextChunk] = field(default_factory=list)
    image_chunks: List[PDFImageChunk] = field(default_factory=list)


def parse_pdf_bytes(file_bytes: bytes, filename: str = "document.pdf") -> ParsedPDF:
    """Parses PDF bytes, extracting text and embedded images page by page.

    Args:
        file_bytes: Raw bytes of the PDF file.
        filename: Name of the PDF file.

    Returns:
        ParsedPDF containing extracted text chunks and image chunks.
    """
    logger.info("Parsing PDF bytes for file: %s", filename)
    reader = PdfReader(io.BytesIO(file_bytes))
    parsed = ParsedPDF(filename=filename, num_pages=len(reader.pages))

    for page_idx, page in enumerate(reader.pages, start=1):
        # Extract text
        text_content = page.extract_text() or ""
        if text_content.strip():
            # Paragraph or section splitting logic
            paragraphs = [p.strip() for p in text_content.split("\n\n") if p.strip()]
            for p_idx, para in enumerate(paragraphs, start=1):
                chunk_id = f"{filename}_p{page_idx}_t{p_idx}"
                parsed.text_chunks.append(
                    PDFTextChunk(page_number=page_idx, text=para, chunk_id=chunk_id)
                )

        # Extract embedded images
        try:
            for img_idx, img_file in enumerate(page.images, start=1):
                image_id = f"{filename}_p{page_idx}_img{img_idx}"
                img_format = img_file.name.split(".")[-1].upper() if "." in img_file.name else "PNG"
                parsed.image_chunks.append(
                    PDFImageChunk(
                        page_number=page_idx,
                        image_index=img_idx,
                        image_bytes=img_file.data,
                        format=img_format,
                        image_id=image_id,
                    )
                )
        except Exception as err:
            logger.warning("Error extracting images on page %d of %s: %s", page_idx, filename, err)

    logger.info(
        "Parsed %s: %d pages, %d text chunks, %d image chunks.",
        filename,
        parsed.num_pages,
        len(parsed.text_chunks),
        len(parsed.image_chunks),
    )
    return parsed
