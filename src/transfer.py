import io
import json

import pyarrow as pa
import pyarrow.parquet as pq
from sqlalchemy.orm import Session
from sqlalchemy import text

from src.models import DocumentChunk


def export_chunks(db: Session) -> bytes:
    """Export all chunks and embeddings to a Parquet file."""
    results = db.execute(
        text("""
            SELECT source_url, resource_type, title, content,
                   embedding::text AS embedding
            FROM document_chunks
            ORDER BY id
        """)
    ).fetchall()

    table = pa.table({
        "source_url": [r.source_url for r in results],
        "resource_type": [r.resource_type for r in results],
        "title": [r.title for r in results],
        "content": [r.content for r in results],
        "embedding": [json.loads(r.embedding) for r in results],
    })

    buffer = io.BytesIO()
    pq.write_table(table, buffer, compression="zstd")
    return buffer.getvalue()


def import_chunks(db: Session, data: bytes, clear: bool = False) -> dict[str, int]:
    """Import chunks and embeddings from a Parquet file."""
    if clear:
        db.query(DocumentChunk).delete()
        db.commit()

    buffer = io.BytesIO(data)
    table = pq.read_table(buffer)

    imported = 0
    for i in range(table.num_rows):
        doc = DocumentChunk(
            source_url=table.column("source_url")[i].as_py(),
            resource_type=table.column("resource_type")[i].as_py(),
            title=table.column("title")[i].as_py(),
            content=table.column("content")[i].as_py(),
            embedding=table.column("embedding")[i].as_py(),
        )
        db.add(doc)
        imported += 1

        if imported % 500 == 0:
            db.commit()

    db.commit()

    return {
        "chunks_imported": imported,
    }
