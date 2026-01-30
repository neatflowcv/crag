from crag.models.state import CRAGState
from crag.vectorstore.store import VectorStore


def retrieve(state: CRAGState, store: VectorStore) -> CRAGState:
    search_query = state.get("search_query") or state["question"]
    documents = store.search(search_query)

    return {
        **state,
        "documents": documents,
    }
