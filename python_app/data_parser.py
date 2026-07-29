from dataclasses import dataclass
from pathlib import Path

import fitz


class PDFError(ValueError):
    """Base exception for invalid or unreadable PDFs."""


class ScannedPDFError(PDFError):
    """Raised when a PDF appears to be image-only."""


@dataclass(frozen=True)
class ParsedPage:
    page_number: int
    text: str


def validate_pdf_path(pdf_path: str) -> None:
    path = Path(pdf_path)
    if path.suffix.lower() != ".pdf":
        raise PDFError("Only PDF files are supported.")
    if not path.is_file():
        raise PDFError("The uploaded PDF could not be found.")


def parse_pdf_pages(
    pdf_path: str,
    min_chars_per_text_page: int = 30,
) -> list[ParsedPage]:
    """Extract text page-by-page and reject encrypted or image-only PDFs."""
    validate_pdf_path(pdf_path)

    try:
        with fitz.open(pdf_path) as doc:
            if doc.needs_pass:
                raise PDFError("Password-protected PDFs are not supported.")

            pages = [
                ParsedPage(page_number=index + 1, text=page.get_text("text").strip())
                for index, page in enumerate(doc)
            ]
    except PDFError:
        raise
    except Exception as exc:
        raise PDFError(f"Could not read this PDF: {exc}") from exc

    if not pages:
        raise PDFError("The PDF contains no pages.")

    if not any(len(page.text) >= min_chars_per_text_page for page in pages):
        raise ScannedPDFError(
            "This PDF contains little or no embedded text and appears to be scanned or image-only."
        )

    return pages


def parse_pdf(pdf_path: str) -> str:
    """Backward-compatible helper returning all extracted text."""
    return "\n\n".join(page.text for page in parse_pdf_pages(pdf_path) if page.text)
