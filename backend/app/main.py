from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routers import ingest, chat

app = FastAPI(title="AI Research Copilot (Open-Source)", version="0.1.0")

# ✅ Enable CORS so frontend (http://127.0.0.1:5173) can call backend (http://127.0.0.1:8000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # for dev, allow all. tighten in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"ok": True, "service": "ai-research-copilot-backend"}

# Routers
app.include_router(ingest.router)
app.include_router(chat.router)
