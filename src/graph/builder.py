from typing import Literal

from langgraph.graph import END, StateGraph

from src.config.settings import settings
from src.graph.nodes.generator import generate
from src.graph.nodes.grader import grade_documents
from src.graph.nodes.html_fetcher import fetch_html
from src.graph.nodes.query_rewriter import rewrite_query
from src.graph.nodes.retriever import retrieve
from src.graph.nodes.web_searcher import web_search
from src.models.state import CRAGState


def should_continue(state: CRAGState) -> Literal["generate", "rewrite_query"]:
    if state.get("documents_relevant", False):
        return "generate"

    retry_count = state.get("retry_count", 0)
    if retry_count >= settings.max_retries:
        return "generate"

    return "rewrite_query"


def build_graph() -> StateGraph:
    workflow = StateGraph(CRAGState)

    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("generate", generate)
    workflow.add_node("rewrite_query", rewrite_query)
    workflow.add_node("web_search", web_search)
    workflow.add_node("fetch_html", fetch_html)

    workflow.set_entry_point("retrieve")

    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        should_continue,
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
