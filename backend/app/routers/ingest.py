import os
import tempfile
from pathlib import Path
from typing import List, Dict, Any

from fastapi import APIRouter, UploadFile, Depends, HTTPException
from sqlalchemy.orm import Session

from ..db import SessionLocal, Document, init_db
from ..chunking import read_pdf_text, chunk_text
from ..rag import Embedder
from ..deps import settings

# Text cleanup
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


# Provider-aware embedder (reads env)
embedder = Embedder()
if embedder.dim != settings.EMBED_DIM:
    raise RuntimeError(f"EMBED_DIM={settings.EMBED_DIM} but model returns {embedder.dim}")

# Ensure tables/extension exist (idempotent)
init_db()


def clean_text(t: str) -> str:
    """
    Normalize PDF text for better embeddings and chunking.
    """
    if not t:
        return ""
    t = fix_text(t)
    t = unidecode(t)  # comment out if you prefer curly quotes/dashes
    # de-hyphenate line wraps: "trans-\nform" -> "transform"
    t = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", t)
    # soft line breaks -> space (preserve paragraph breaks)
    t = re.sub(r"(?<!\n)\n(?!\n)", " ", t)
    # collapse whitespace
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


@router.post("/pdf/{doc_id}")
async def ingest_pdf(doc_id: str, file: UploadFile, db: Session = Depends(get_db)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    # Save to a temp file cross-platform
    tmp_dir = Path(tempfile.gettempdir())
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = tmp_dir / f"{doc_id}.pdf"

    try:
        with open(tmp_path, "wb") as f:
            f.write(await file.read())

        # Extract + normalize text
        raw_text = read_pdf_text(str(tmp_path))
    finally:
        # best effort cleanup
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass

    if not raw_text or not raw_text.strip():
        raise HTTPException(status_code=400, detail="No text extracted from PDF.")

    text = clean_text(raw_text)

    # Chunk & embed (embedder should return numpy arrays or lists)
    chunks: List[str] = chunk_text(text)
    if not chunks:
        raise HTTPException(status_code=400, detail="No chunks produced from text.")

    vecs = embedder.embed(chunks)  # expect len(vecs) == len(chunks)

    if len(vecs) != len(chunks):
        raise HTTPException(status_code=500, detail="Embedding count mismatch with chunks.")

    # Prepare rows; JSONB wants dict (not string); pgvector wants list[float]
    rows: List[Document] = []
    for i, (chunk, vec) in enumerate(zip(chunks, vecs)):
        # vec can be list or np.ndarray; ensure list[float]
        try:
            embedding_vec = vec.tolist() if hasattr(vec, "tolist") else list(vec)
        except Exception:
            raise HTTPException(status_code=500, detail="Embedding must be list-like of floats.")

        rows.append(
            Document(
                doc_id=doc_id,
                chunk_id=i,
                content=chunk,
                meta={},                 # ✅ JSONB dict (not a string); use {} or your own dict
                embedding=embedding_vec  # ✅ list[float], correct dim
            )
        )

    # Insert in a transaction
    try:
        db.bulk_save_objects(rows)  # faster than many adds
        db.commit()
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"DB insert failed: {e!s}")

    return {"ok": True, "doc_id": doc_id, "chunks": len(rows)}
