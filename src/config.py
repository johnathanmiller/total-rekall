from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str = "postgresql://rekall:rekall@db:5432/rekall"
    embedding_model: str = "all-MiniLM-L6-v2"
    ollama_url: str = "http://host.docker.internal:11434"
    ollama_model: str = "llama3.1:8b"
    embedding_dimensions: int = 384
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k: int = 5

    class Config:
        env_file = ".env"


settings = Settings()
