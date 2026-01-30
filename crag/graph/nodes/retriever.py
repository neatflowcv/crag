from crag.models.state import CRAGState
from crag.vectorstore.store import VectorStore


def retrieve(state: CRAGState, store: VectorStore) -> CRAGState:
    search_query = state.get("search_query") or state["question"]
    documents = store.search(search_query)

    print(f"[Retrieve] '{search_query}' -> {len(documents)} documents")

    return {
        **state,
        "documents": documents,
    }
