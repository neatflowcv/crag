from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Ollama settings
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3:1.7b"

    # ChromaDB settings
    chroma_persist_dir: Path = Path("data/chroma_db")
    chroma_collection_name: str = "crag_documents"

    # Embedding model
    embedding_model: str = "intfloat/multilingual-e5-small"
    embedding_model_local_dir: Path = Path("models/multilingual-e5-small")

    # RAG settings
    retriever_k: int = 4
    max_retries: int = 2


settings = Settings()
