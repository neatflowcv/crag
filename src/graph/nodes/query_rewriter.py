from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from src.config.settings import settings
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


def rewrite_query(state: CRAGState) -> CRAGState:
    question = state["question"]

    llm = ChatOllama(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
    )

    chain = REWRITER_PROMPT | llm
    response = chain.invoke({"question": question})

    rewritten_query = strip_think_tags(response.content)

    return {
        **state,
        "search_query": rewritten_query,
        "retry_count": state.get("retry_count", 0) + 1,
    }
