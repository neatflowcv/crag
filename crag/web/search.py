from abc import ABC, abstractmethod
from dataclasses import dataclass

import httpx


@dataclass
class SearchResult:
    title: str
    href: str
    body: str

    def to_dict(self) -> dict:
        return {"title": self.title, "href": self.href, "body": self.body}


class WebSearchStrategy(ABC):
    @abstractmethod
    def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        pass


class DuckDuckGoInstantAnswerStrategy(WebSearchStrategy):
    """DuckDuckGo Instant Answer API (위키피디아 기반)."""

    def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        response = httpx.get(
            "https://api.duckduckgo.com/",
            params={"q": query, "format": "json", "no_html": "1"},
            timeout=10.0,
        )
        data = response.json()

        results: list[SearchResult] = []

        if data.get("Abstract"):
            results.append(
                SearchResult(
                    title=data.get("Heading", ""),
                    href=data.get("AbstractURL", ""),
                    body=data.get("Abstract", ""),
                )
            )

        for topic in data.get("RelatedTopics", []):
            if len(results) >= max_results:
                break
            if isinstance(topic, dict) and "FirstURL" in topic:
                results.append(
                    SearchResult(
                        title=topic.get("Text", "")[:100],
                        href=topic.get("FirstURL", ""),
                        body=topic.get("Text", ""),
                    )
                )

        return results


class DuckDuckGoTextSearchStrategy(WebSearchStrategy):
    """DuckDuckGo Text Search (duckduckgo-search 패키지)."""

    def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        from duckduckgo_search import DDGS

        with DDGS() as ddgs:
            raw_results = list(ddgs.text(query, max_results=max_results))

        return [
            SearchResult(
                title=r.get("title", ""),
                href=r.get("href", ""),
                body=r.get("body", ""),
            )
            for r in raw_results
        ]


class LocalSearchStrategy(WebSearchStrategy):
    """로컬 검색 서버 (http://127.0.0.1:5000)."""

    def __init__(self, base_url: str = "http://127.0.0.1:5000") -> None:
        self._base_url = base_url

    def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        response = httpx.get(
            f"{self._base_url}/search",
            params={"q": query, "format": "json"},
            timeout=30.0,
        )
        data = response.json()

        results: list[SearchResult] = []
        for item in data.get("results", [])[:max_results]:
            results.append(
                SearchResult(
                    title=item.get("title", ""),
                    href=item.get("href", ""),
                    body=item.get("content", ""),
                )
            )

        return results


class WebSearcher:
    def __init__(self, strategy: WebSearchStrategy) -> None:
        self._strategy = strategy

    def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        return self._strategy.search(query, max_results)


# 기본 검색기 (하위 호환성)
_default_searcher = WebSearcher(LocalSearchStrategy())


def search_web(query: str, max_results: int = 5) -> list[dict]:
    """하위 호환성을 위한 함수."""
    results = _default_searcher.search(query, max_results)
    return [r.to_dict() for r in results]
