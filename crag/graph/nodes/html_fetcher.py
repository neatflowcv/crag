import logging

import httpx
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel

from crag.models.state import CRAGState
from crag.vectorstore.store import VectorStore

logger = logging.getLogger(__name__)

HTML_TO_MARKDOWN_PROMPT = """Convert the following HTML content to clean Markdown.
Extract only the main content, removing navigation, ads, footers, and other irrelevant elements.
Keep the structure with headings, lists, and code blocks where appropriate.
If the content is not in English, keep the original language.

HTML:
{html}

Markdown:"""


def _fetch_html(url: str) -> str | None:
    """URL에서 HTML을 가져온다."""
    try:
        response = httpx.get(
            url,
            timeout=15.0,
            follow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0 (compatible; CRAG/1.0)"},
        )
        response.raise_for_status()
        return response.text
    except (httpx.RequestError, httpx.HTTPStatusError) as e:
        logger.warning("[Fetch HTML] Failed to fetch %s: %s", url, e)
        return None


def _html_to_markdown(html: str, llm: BaseChatModel) -> str:
    """LLM을 사용해 HTML을 Markdown으로 변환한다."""
    # HTML이 너무 길면 truncate (토큰 제한)
    max_chars = 30000
    if len(html) > max_chars:
        html = html[:max_chars] + "\n... (truncated)"

    prompt = HTML_TO_MARKDOWN_PROMPT.format(html=html)
    response = llm.invoke(prompt)
    return response.content


def fetch_html(
    state: CRAGState, store: VectorStore, llm: BaseChatModel
) -> CRAGState:
    """웹 검색 결과 URL에서 HTML을 가져와 Markdown으로 변환 후 저장한다."""
    web_results = state.get("web_search_results", [])
    documents = list(state.get("documents", []))

    for result in web_results:
        url = result.get("href", "")
        if not url:
            continue

        # 이미 저장된 URL은 스킵
        if store.exists_by_source(url):
            logger.debug("[Fetch HTML] Already exists: %s", url)
            continue

        # HTML 가져오기
        html = _fetch_html(url)
        if not html:
            continue

        # LLM으로 Markdown 변환
        logger.info("[Fetch HTML] Converting to markdown: %s", url)
        markdown = _html_to_markdown(html, llm)

        doc = Document(
            page_content=markdown,
            metadata={
                "source": url,
                "title": result.get("title", ""),
            },
        )
        documents.append(doc)
        store.add_documents([doc])

    return {
        **state,
        "documents": documents,
    }
