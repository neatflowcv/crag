import logging

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

from crag.models.state import CRAGState
from crag.utils import strip_think_tags

logger = logging.getLogger(__name__)

REWRITER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a web search query generator.
Given a user's question and the reason why web search is needed, generate 3 to 10 effective English search queries.
Focus on what information is missing based on the reason.
Each query should capture different aspects or phrasings.
Output one query per line, nothing else.""",
        ),
        (
            "human",
            """Question: {question}

Reason for web search: {reason}

Search queries (one per line):""",
        ),
    ]
)


def rewrite_query(state: CRAGState, llm: BaseChatModel) -> CRAGState:
    question = state["question"]
    reason = state.get("web_search_reason", "")

    chain = REWRITER_PROMPT | llm
    response = chain.invoke({"question": question, "reason": reason})

    raw_queries = strip_think_tags(response.content)
    queries = [q.strip() for q in raw_queries.strip().split("\n") if q.strip()]

    # 3개 미만이면 원본 질문 추가
    if len(queries) < 3:
        queries.append(question)

    # 10개 초과면 자르기
    queries = queries[:10]

    logger.info("[Web Search] Reason: %s", reason)
    logger.info("[Web Search] Queries:")
    for i, q in enumerate(queries, 1):
        logger.info("  %d. %s", i, q)

    return {
        **state,
        "web_search_queries": queries,
        "retry_count": state.get("retry_count", 0) + 1,
    }
