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
            SELECT id, resource_type, title, content, source_url, chunk_index,
                   embedding <=> CAST(:embedding AS vector) AS distance
            FROM document_chunks
            ORDER BY embedding <=> CAST(:embedding AS vector)
            LIMIT :limit
        """),
        {"embedding": str(query_embedding), "limit": k},
    ).fetchall()

    matched = [
        {
            "id": row.id,
            "resource_type": row.resource_type,
            "title": row.title,
            "content": row.content,
            "source_url": row.source_url,
            "chunk_index": row.chunk_index,
            "distance": row.distance,
        }
        for row in results
    ]

    return _expand_with_neighbors(db, matched)


def _expand_with_neighbors(db: Session, chunks: list[dict]) -> list[dict]:
    """Fetch chunk_index ± 1 neighbors for each matched chunk, deduplicate, and order."""
    seen_ids = {c["id"] for c in chunks}
    all_chunks = list(chunks)

    neighbor_params = []
    for chunk in chunks:
        neighbor_params.append(
            {"source_url": chunk["source_url"], "idx": chunk["chunk_index"] - 1}
        )
        neighbor_params.append(
            {"source_url": chunk["source_url"], "idx": chunk["chunk_index"] + 1}
        )

    for params in neighbor_params:
        rows = db.execute(
            text("""
                SELECT id, resource_type, title, content, source_url, chunk_index
                FROM document_chunks
                WHERE source_url = :source_url AND chunk_index = :idx
            """),
            params,
        ).fetchall()

        for row in rows:
            if row.id not in seen_ids:
                seen_ids.add(row.id)
                all_chunks.append({
                    "id": row.id,
                    "resource_type": row.resource_type,
                    "title": row.title,
                    "content": row.content,
                    "source_url": row.source_url,
                    "chunk_index": row.chunk_index,
                    "distance": None,
                })

    all_chunks.sort(key=lambda c: (c["source_url"], c["chunk_index"]))
    return all_chunks


def build_context(chunks: list[dict]) -> str:
    # Group by source_url, chunks are already sorted by (source_url, chunk_index)
    groups: dict[str, list[dict]] = {}
    for chunk in chunks:
        groups.setdefault(chunk["source_url"], []).append(chunk)

    context_parts = []
    for source_url, group in groups.items():
        title = group[0]["title"]
        content = "\n".join(c["content"] for c in group)
        context_parts.append(
            f"--- {title} ---\n"
            f"Source: {source_url}\n\n"
            f"{content}\n"
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
    scored = [c for c in chunks if c.get("distance") is not None]
    avg_distance = sum(c["distance"] for c in scored) / len(scored) if scored else 1.0
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
