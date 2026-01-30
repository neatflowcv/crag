from typing import TypedDict

from langchain_core.documents import Document


class CRAGState(TypedDict):
    question: str
    search_query: str
    documents: list[Document]
    web_search_results: list[dict]
    generation: str
    retry_count: int
    documents_relevant: bool
