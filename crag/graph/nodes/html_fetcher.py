from langchain_core.documents import Document

from crag.models.state import CRAGState
from crag.vectorstore.store import VectorStore


def fetch_html(state: CRAGState, store: VectorStore) -> CRAGState:
    web_results = state.get("web_search_results", [])
    documents = list(state.get("documents", []))

    for result in web_results:
        body = result.get("body", "")
        if body:
            doc = Document(
                page_content=body,
                metadata={
                    "source": result.get("href", ""),
                    "title": result.get("title", ""),
                },
            )
            documents.append(doc)

    new_docs = [
        doc
        for doc in documents
        if doc.metadata.get("source", "").startswith("http")
        and not store.exists_by_source(doc.metadata["source"])
    ]
    if new_docs:
        store.add_documents(new_docs)

    return {
        **state,
        "documents": documents,
    }
