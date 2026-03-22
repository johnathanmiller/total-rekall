from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from src.config import settings
from src.database import init_db, get_db
from src.ingestion import ingest_url
from src.retrieval import query_rag


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()

    # Pull the Ollama model in the background if not already available
    try:
        tags = httpx.get(f"{settings.ollama_url}/api/tags", timeout=5.0).json()
        model_names = [m["name"] for m in tags.get("models", [])]
        if settings.ollama_model not in model_names and f"{settings.ollama_model}:latest" not in model_names:
            print(f"Pulling Ollama model '{settings.ollama_model}'... this may take a few minutes.")
            httpx.post(
                f"{settings.ollama_url}/api/pull",
                json={"name": settings.ollama_model, "stream": False},
                timeout=600.0,
            )
            print(f"Model '{settings.ollama_model}' ready.")
        else:
            print(f"Model '{settings.ollama_model}' already available.")
    except Exception as e:
        print(f"Warning: Could not pull Ollama model '{settings.ollama_model}': {e}. Pull it manually.")

    yield


app = FastAPI(
    title="total-rekall",
    description="RAG system for querying documentation using pgvector and Ollama",
    version="0.1.0",
    lifespan=lifespan,
)


class IngestRequest(BaseModel):
    url: str
    depth: int = 1
    clear: bool = False


class IngestResponse(BaseModel):
    pages_ingested: int
    total_chunks: int
    pages_found: int


class QueryRequest(BaseModel):
    question: str
    top_k: int | None = None


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
    chunks_used: int
    relevance: str
    avg_distance: float


static_dir = Path(__file__).parent / "static"


@app.get("/")
def index():
    return FileResponse(static_dir / "index.html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ingest", response_model=IngestResponse)
def ingest(request: IngestRequest, db: Session = Depends(get_db)):
    result = ingest_url(db, url=request.url, depth=request.depth, clear=request.clear)
    return IngestResponse(**result)


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest, db: Session = Depends(get_db)):
    result = query_rag(db, request.question)
    return QueryResponse(**result)
