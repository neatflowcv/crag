from crag.models.state import CRAGState
from crag.web.search import WebSearchStrategy


def web_search(state: CRAGState, strategy: WebSearchStrategy) -> CRAGState:
    search_queries = state.get("web_search_queries") or [state["question"]]

    # 여러 검색어로 웹 검색하고 결과 합치기 (URL 기준 중복 제거)
    seen_urls: set[str] = set()
    all_results: list[dict] = []

    for query in search_queries:
        results = strategy.search(query, max_results=3)
        new_count = 0
        for result in results:
            url = result.href
            if url and url not in seen_urls:
                seen_urls.add(url)
                all_results.append(result.to_dict())
                new_count += 1
        print(f"[Web Search] '{query}' -> {len(results)} results ({new_count} new)")

    print(f"[Web Search] Total: {len(all_results)} unique results")

    return {
        **state,
        "web_search_results": all_results,
    }
