import logging

from crag.models.state import CRAGState
from crag.vectorstore.store import VectorStore

logger = logging.getLogger(__name__)


def retrieve(state: CRAGState, store: VectorStore) -> CRAGState:
    search_query = state.get("search_query") or state["question"]
    documents = store.search(search_query)

    logger.info("[Retrieve] '%s' -> %d documents", search_query, len(documents))

    return {
        **state,
        "documents": documents,
    }
