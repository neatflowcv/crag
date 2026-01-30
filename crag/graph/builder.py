from typing import Literal

from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, StateGraph

from crag.config.settings import settings
from crag.graph.nodes.generator import generate
from crag.graph.nodes.grader import grade_documents
from crag.graph.nodes.html_fetcher import fetch_html
from crag.graph.nodes.query_rewriter import rewrite_query
from crag.graph.nodes.retriever import retrieve
from crag.graph.nodes.web_searcher import web_search
from crag.models.state import CRAGState
from crag.vectorstore.store import VectorStore


def should_continue(state: CRAGState) -> Literal["generate", "rewrite_query"]:
    if state.get("documents_relevant", False):
        return "generate"

    retry_count = state.get("retry_count", 0)
    if retry_count >= settings.max_retries:
        return "generate"

    return "rewrite_query"


def build_graph(llm: BaseChatModel, store: VectorStore) -> StateGraph:
    workflow = StateGraph(CRAGState)

    workflow.add_node("retrieve", lambda s: retrieve(s, store))
    workflow.add_node("grade_documents", lambda s: grade_documents(s, llm))
    workflow.add_node("generate", lambda s: generate(s, llm))
    workflow.add_node("rewrite_query", lambda s: rewrite_query(s, llm))
    workflow.add_node("web_search", web_search)
    workflow.add_node("fetch_html", lambda s: fetch_html(s, store))

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
