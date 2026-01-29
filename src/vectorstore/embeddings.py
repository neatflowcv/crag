from sentence_transformers import SentenceTransformer

from src.config.settings import settings


def get_embedding_model() -> SentenceTransformer:
    return SentenceTransformer(settings.embedding_model)
