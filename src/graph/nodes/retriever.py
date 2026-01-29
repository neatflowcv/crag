from src.models.state import CRAGState
from src.vectorstore.store import VectorStore


def retrieve(state: CRAGState) -> CRAGState:
    search_query = state.get("search_query") or state["question"]
    store = VectorStore()
    documents = store.search(search_query)

    return {
        **state,
        "documents": documents,
    }
