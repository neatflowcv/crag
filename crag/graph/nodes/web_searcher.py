from crag.models.state import CRAGState
from crag.web.search import search_web


def web_search(state: CRAGState) -> CRAGState:
    search_query = state.get("search_query") or state["question"]
    results = search_web(search_query, max_results=5)

    return {
        **state,
        "web_search_results": results,
    }
