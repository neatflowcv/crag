from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

from src.models.state import CRAGState
from src.utils import strip_think_tags

REWRITER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a query rewriter. Rewrite the question in English as a short search query (2-5 words).
Output only the query, nothing else.""",
        ),
        (
            "human",
            """Question: {question}

English search query:""",
        ),
    ]
)


def rewrite_query(state: CRAGState, llm: BaseChatModel) -> CRAGState:
    question = state["question"]

    chain = REWRITER_PROMPT | llm
    response = chain.invoke({"question": question})

    rewritten_query = strip_think_tags(response.content)

    return {
        **state,
        "search_query": rewritten_query,
        "retry_count": state.get("retry_count", 0) + 1,
    }
