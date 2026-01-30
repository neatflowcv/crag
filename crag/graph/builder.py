from typing import Literal

from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, StateGraph

from crag.config.settings import settings
from crag.graph.nodes.generator import generate
from crag.graph.nodes.grader import grade_documents
from crag.graph.nodes.html_fetcher import fetch_html
from crag.graph.nodes.query_rewriter import rewrite_query
from crag.graph.nodes.retriever import retrieve
from crag.graph.nodes.web_search_decision import decide_web_search
from crag.graph.nodes.web_searcher import web_search
from crag.models.state import CRAGState
from crag.vectorstore.store import VectorStore


def after_grade(state: CRAGState) -> Literal["generate", "decide_web_search"]:
    """문서 관련성 판단 후 다음 단계 결정."""
    if state.get("documents_relevant", False):
        return "generate"
    return "decide_web_search"


def after_web_search_decision(
    state: CRAGState,
) -> Literal["generate", "rewrite_query"]:
    """웹 서치 필요성 판단 후 다음 단계 결정."""
    retry_count = state.get("retry_count", 0)
    if retry_count >= settings.max_retries:
        return "generate"

    if state.get("needs_web_search", False):
        return "rewrite_query"
    return "generate"


def build_graph(llm: BaseChatModel, store: VectorStore) -> StateGraph:
    workflow = StateGraph(CRAGState)

    workflow.add_node("retrieve", lambda s: retrieve(s, store))
    workflow.add_node("grade_documents", lambda s: grade_documents(s, llm))
    workflow.add_node("decide_web_search", lambda s: decide_web_search(s, llm))
    workflow.add_node("generate", lambda s: generate(s, llm))
    workflow.add_node("rewrite_query", lambda s: rewrite_query(s, llm))
    workflow.add_node("web_search", web_search)
    workflow.add_node("fetch_html", lambda s: fetch_html(s, store))

    workflow.set_entry_point("retrieve")

    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        after_grade,
        {
            "generate": "generate",
            "decide_web_search": "decide_web_search",
        },
    )
    workflow.add_conditional_edges(
        "decide_web_search",
        after_web_search_decision,
        {
            "generate": "generate",
            "rewrite_query": "rewrite_query",
        },
    )
    workflow.add_edge("rewrite_query", "web_search")
    workflow.add_edge("web_search", "fetch_html")
    workflow.add_edge("fetch_html", "retrieve")
    workflow.add_edge("generate", END)

    return workflow.compile()
