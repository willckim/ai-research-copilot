# backend/app/rag.py
import asyncio
import logging
from typing import Optional

import httpx
import numpy as np
from sentence_transformers import SentenceTransformer

from .deps import settings

logger = logging.getLogger(__name__)

# =========================
# Embeddings (provider-aware)
# =========================

try:
    from openai import OpenAI            # installed via `pip install openai`
except Exception:                         # tolerate missing pkg in local-only mode
    OpenAI = None                         # type: ignore


class Embedder:
    """
    Provider-aware embedder.
    - local  -> sentence-transformers (e.g., all-MiniLM-L6-v2, 384-dim)
    - openai -> text-embedding-3-*, 1536-dim (or per EMBED_DIM)
    Returns numpy arrays shaped (n, dim).
    """
    def __init__(self):
        self.kind = (settings.EMBED_PROVIDER or "local").lower()
        if self.kind == "local":
            self.model = SentenceTransformer(settings.EMBED_MODEL)
            self.dim = self.model.get_sentence_embedding_dimension()
        elif self.kind == "openai":
            if OpenAI is None:
                raise RuntimeError("openai package not installed but EMBED_PROVIDER=openai")
            self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
            self.dim = settings.EMBED_DIM
        else:
            raise ValueError(f"Unsupported EMBED_PROVIDER: {settings.EMBED_PROVIDER}")

    def embed(self, texts: list[str]) -> np.ndarray:
        if self.kind == "local":
            # normalize for cosine distance in pgvector
            return self.model.encode(texts, normalize_embeddings=True)
        else:
            resp = self.client.embeddings.create(model=settings.EMBED_MODEL, input=texts)
            vecs = [d.embedding for d in resp.data]
            return np.asarray(vecs, dtype=float)


# =========================
# LLM (provider-aware)
# =========================

# Reuse one Async HTTP client for Ollama
TIMEOUT = httpx.Timeout(connect=5.0, read=300.0, write=60.0, pool=5.0)
LIMITS = httpx.Limits(max_keepalive_connections=5, max_connections=10)
_client: Optional[httpx.AsyncClient] = None


def _client_get() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(timeout=TIMEOUT, limits=LIMITS)
    return _client


async def call_llm(prompt: str) -> str:
    """
    Provider-aware text generation.
    - ollama: POST /api/generate
    - openai: chat.completions.create (run in a worker thread)
    Returns plain text or a marker starting with:
      [LLM timeout] / [LLM connection error] / [LLM error]
    """
    prov = (settings.LLM_PROVIDER or "ollama").lower()

    # ---------- OLLAMA ----------
    if prov == "ollama":
        payload = {
            "model": settings.LLM_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.2, "num_predict": settings.LLM_NUM_PREDICT},
        }
        url = f"{settings.OLLAMA_URL}/api/generate"
        logger.info("Calling Ollama generate", extra={"url": url, "model": settings.LLM_MODEL})

        client = _client_get()
        try:
            r = await client.post(url, json=payload)
            r.raise_for_status()
            data = r.json()
            return (data.get("response") or "").strip()
        except (httpx.ReadTimeout, httpx.WriteTimeout, httpx.TimeoutException):
            logger.exception("LLM timeout")
            return "[LLM timeout] The model did not respond in time."
        except httpx.HTTPStatusError as e:
            logger.exception("LLM HTTP status error")
            return f"[LLM error] {e.response.status_code}: {e.response.text}"
        except httpx.RequestError as e:
            logger.exception("LLM connection/request error")
            return f"[LLM connection error] {e}"

    # ---------- OPENAI ----------
    try:
        if OpenAI is None:
            return "[LLM error] openai package not installed"
        client = OpenAI(api_key=settings.OPENAI_API_KEY)

        def _run():
            resp = client.chat.completions.create(
                model=settings.OPENAI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=settings.LLM_MAX_TOKENS,
                temperature=0.2,
            )
            return resp.choices[0].message.content.strip()

        # OpenAI SDK call is sync; run it off the event loop
        return await asyncio.to_thread(_run)

    except Exception as e:
        logger.exception("OpenAI call failed")
        return f"[LLM error] {e}"


# Backward-compat: old code may still import call_mistral
async def call_mistral(prompt: str) -> str:
    return await call_llm(prompt)
