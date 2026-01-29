from src.models.state import CRAGState
from src.web.search import search_web


def web_search(state: CRAGState) -> CRAGState:
    question = state["question"]

    results = search_web(question, max_results=5)
    urls = [result["href"] for result in results if "href" in result]

    return {
        **state,
        "web_search_urls": urls,
    }
