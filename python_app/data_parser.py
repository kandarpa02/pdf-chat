import fitz


class ScannedPDFError(Exception):
    """Raised when the uploaded PDF appears to be image-only."""
    pass


def validate_digital_pdf(
    pdf_path: str,
    min_chars_per_page: int = 30,
    min_text_pages: int = 1,
) -> None:
    """
    Raises ScannedPDFError if the PDF appears to be scanned/image-only.
    """

    text_pages = 0

    with fitz.open(pdf_path) as doc:
        for page in doc:
            if len(page.get_text()) >= min_chars_per_page:
                text_pages += 1

                # Early exit
                if text_pages >= min_text_pages:
                    return

    raise ScannedPDFError(
        "This PDF appears to contain little or no embedded text and is likely scanned or image-only."
    )


def parse_pdf(pdf_path: str) -> str:
    """
    Validates that the PDF contains embedded text and returns the
    extracted text as a single string.

    Args:
        pdf_path: Path to the PDF.

    Returns:
        Full extracted text.

    Raises:
        ScannedPDFError: If the PDF is scanned/image-only.
    """

    validate_digital_pdf(pdf_path)

    with fitz.open(pdf_path) as doc:
        text = "\n".join(page.get_text() for page in doc)

    return text
