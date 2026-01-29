from src.models.state import CRAGState
from src.vectorstore.store import VectorStore


def retrieve(state: CRAGState) -> CRAGState:
    question = state["question"]
    store = VectorStore()
    documents = store.search(question)

    return {
        **state,
        "documents": documents,
    }
