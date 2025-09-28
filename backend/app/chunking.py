from pypdf import PdfReader


def read_pdf_text(path: str) -> str:
    """
    Extract all text from a PDF file.

    Args:
        path (str): Path to the PDF.

    Returns:
        str: Concatenated text of all pages.
    """
    reader = PdfReader(path)
    texts = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            texts.append(text.strip())
    return "\n".join(texts)


def chunk_text(text: str, chunk_size: int = 300, chunk_overlap: int = 50) -> list[str]:
    """
    Split text into overlapping word chunks for embedding.

    Args:
        text (str): The full document text.
        chunk_size (int): Number of words per chunk.
        chunk_overlap (int): How many words overlap between chunks.

    Returns:
        list[str]: List of chunk strings.
    """
    words = text.split()
    if not words:
        return []

    chunks = []
    i = 0
    step = max(1, chunk_size - chunk_overlap)

    while i < len(words):
        chunk = words[i : i + chunk_size]
        chunks.append(" ".join(chunk))
        i += step

    return chunks
