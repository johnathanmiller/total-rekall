import httpx
from sqlalchemy.orm import Session
from sqlalchemy import text

from src.config import settings
from src.embeddings import generate_embedding


def search_similar_chunks(db: Session, query: str, top_k: int | None = None) -> list[dict]:
    k = top_k or settings.top_k
    query_embedding = generate_embedding(query)

    results = db.execute(
        text("""
            SELECT id, resource_type, title, content, source_url,
                   embedding <=> CAST(:embedding AS vector) AS distance
            FROM document_chunks
            ORDER BY embedding <=> CAST(:embedding AS vector)
            LIMIT :limit
        """),
        {"embedding": str(query_embedding), "limit": k},
    ).fetchall()

    return [
        {
            "id": row.id,
            "resource_type": row.resource_type,
            "title": row.title,
            "content": row.content,
            "source_url": row.source_url,
            "distance": row.distance,
        }
        for row in results
    ]


def build_context(chunks: list[dict]) -> str:
    context_parts = []
    for chunk in chunks:
        context_parts.append(
            f"--- {chunk['title']} ---\n"
            f"Source: {chunk['source_url']}\n\n"
            f"{chunk['content']}\n"
        )
    return "\n".join(context_parts)


def query_rag(db: Session, question: str) -> dict:
    chunks = search_similar_chunks(db, question)
    context = build_context(chunks)

    prompt = (
        "Answer the question using only the provided documentation context. "
        "If the context doesn't contain enough information to answer, say so. "
        "Be specific and reference relevant details from the documentation.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}"
    )

    response = httpx.post(
        f"{settings.ollama_url}/api/generate",
        json={
            "model": settings.ollama_model,
            "prompt": prompt,
            "stream": False,
        },
        timeout=300.0,
    )
    response.raise_for_status()
    answer = response.json()["response"]

    sources = list({chunk["source_url"] for chunk in chunks})
    avg_distance = sum(c["distance"] for c in chunks) / len(chunks) if chunks else 1.0
    # Cosine distance: 0 = identical, 1 = orthogonal, 2 = opposite
    # Below ~0.5 means strong match, above ~0.8 means weak/no match
    relevance = "high" if avg_distance < 0.5 else "medium" if avg_distance < 0.7 else "low"

    return {
        "answer": answer,
        "sources": sources,
        "chunks_used": len(chunks),
        "relevance": relevance,
        "avg_distance": round(avg_distance, 4),
    }
