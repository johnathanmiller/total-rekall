from sentence_transformers import SentenceTransformer

from src.config import settings

_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(settings.embedding_model)
    return _model


def generate_embedding(text: str) -> list[float]:
    model = get_model()
    return model.encode(text).tolist()


def generate_embeddings(texts: list[str]) -> list[list[float]]:
    model = get_model()
    return model.encode(texts).tolist()
