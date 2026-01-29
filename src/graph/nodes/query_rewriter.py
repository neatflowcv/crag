from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from src.config.settings import settings
from src.models.state import CRAGState

REWRITER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a query rewriter that improves search queries.
Rewrite the given question to be more specific and suitable for web search.
Output only the rewritten query, nothing else.""",
        ),
        (
            "human",
            """Original question: {question}

Rewritten query:""",
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

    rewritten_query = response.content.strip()

    return {
        **state,
        "question": rewritten_query,
        "retry_count": state.get("retry_count", 0) + 1,
    }
