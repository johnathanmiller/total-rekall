from sqlalchemy import Column, Integer, String, Text
from pgvector.sqlalchemy import Vector

from src.config import settings
from src.database import Base


class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    source_url = Column(String, nullable=False)
    resource_type = Column(String, nullable=False, index=True)
    title = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    embedding = Column(Vector(settings.embedding_dimensions), nullable=False)
