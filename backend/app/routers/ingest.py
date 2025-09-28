import os
import tempfile
from pathlib import Path
from fastapi import APIRouter, UploadFile, Depends, HTTPException
from sqlalchemy.orm import Session

from ..db import SessionLocal, Document, init_db
from ..chunking import read_pdf_text, chunk_text
from ..rag import Embedder                      # ✅ no settings arg now
from ..deps import settings

# NEW: imports for text cleanup
from ftfy import fix_text
from unidecode import unidecode
import re

router = APIRouter(prefix="/ingest", tags=["ingest"])


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ✅ provider-aware embedder (reads EMBED_PROVIDER / EMBED_MODEL from env)
embedder = Embedder()
# ✅ sanity check in case EMBED_DIM and model dimension differ
if embedder.dim != settings.EMBED_DIM:
    raise RuntimeError(f"EMBED_DIM={settings.EMBED_DIM} but model returns {embedder.dim}")

# Ensure tables exist
init_db()


# NEW: text normalization util
def clean_text(t: str) -> str:
    """
    Fix mojibake/encoding artifacts and normalize PDF text for better embeddings.
    Safe defaults for English. Remove 'unidecode' if you prefer curly quotes/dashes.
    """
    if not t:
        return ""
    # Fix mis-decoded UTF-8 (e.g., â€™ -> ’)
    t = fix_text(t)

    # Normalize fancy punctuation to ASCII (optional; comment out to keep curly quotes)
    t = unidecode(t)

    # De-hyphenate common line-wrapped words: "trans-\nform" -> "transform"
    t = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", t)

    # Replace single newlines (soft wraps) with spaces; keep paragraph breaks
    t = re.sub(r"(?<!\n)\n(?!\n)", " ", t)

    # Collapse excessive whitespace/newlines
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)

    return t.strip()


@router.post("/pdf/{doc_id}")
async def ingest_pdf(doc_id: str, file: UploadFile, db: Session = Depends(get_db)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    # ✅ Cross-platform temp file
    tmp_dir = Path(tempfile.gettempdir())
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = tmp_dir / f"{doc_id}.pdf"

    with open(tmp_path, "wb") as f:
        f.write(await file.read())

    # Extract raw text
    raw_text = read_pdf_text(str(tmp_path))

    # Clean up temp file
    try:
        tmp_path.unlink(missing_ok=True)
    except Exception:
        pass

    if not raw_text or not raw_text.strip():
        raise HTTPException(status_code=400, detail="No text extracted from PDF.")

    # 🔧 Normalize text BEFORE chunking/embedding
    text = clean_text(raw_text)

    # Chunk & embed
    chunks = chunk_text(text)
    vecs = embedder.embed(chunks)

    for i, (chunk, vec) in enumerate(zip(chunks, vecs)):
        row = Document(
            doc_id=doc_id,
            chunk_id=i,
            content=chunk,
            meta=None,               # mapped to DB column "metadata"
            embedding=vec.tolist(),  # pgvector accepts Python lists
        )
        db.add(row)
    db.commit()

    return {"doc_id": doc_id, "chunks": len(chunks)}
