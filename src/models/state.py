from typing import TypedDict

from langchain_core.documents import Document


class CRAGState(TypedDict):
    question: str
    documents: list[Document]
    web_search_urls: list[str]
    generation: str
    retry_count: int
    documents_relevant: bool
