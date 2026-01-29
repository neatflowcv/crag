from langchain_core.documents import Document

from src.models.state import CRAGState
from src.vectorstore.store import VectorStore
from src.web.scraper import fetch_and_parse


def fetch_html(state: CRAGState) -> CRAGState:
    urls = state.get("web_search_urls", [])
    documents = list(state.get("documents", []))

    for url in urls:
        content = fetch_and_parse(url)
        if content:
            doc = Document(page_content=content[:2000], metadata={"source": url})
            documents.append(doc)

    if documents:
        store = VectorStore()
        new_docs = [doc for doc in documents if doc.metadata.get("source", "").startswith("http")]
        if new_docs:
            store.add_documents(new_docs)

    return {
        **state,
        "documents": documents,
    }
