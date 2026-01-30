from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

from crag.models.state import CRAGState
from crag.utils import strip_think_tags

TRANSLATOR_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a query translator. Translate the user's question into English for vector search.
Keep the semantic meaning intact. Output only the translated query, nothing else.""",
        ),
        (
            "human",
            """Question: {question}

English query:""",
        ),
    ]
)


def translate_query(state: CRAGState, llm: BaseChatModel) -> CRAGState:
    """최초 검색을 위해 질문을 영어로 번역."""
    question = state["question"]

    chain = TRANSLATOR_PROMPT | llm
    response = chain.invoke({"question": question})

    translated_query = strip_think_tags(response.content)

    return {
        **state,
        "search_query": translated_query,
    }
