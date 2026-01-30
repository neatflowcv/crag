from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

from crag.models.state import CRAGState
from crag.utils import strip_think_tags

GENERATOR_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful assistant that answers questions based on the provided context.
Use only the information from the context to answer the question.
If the context doesn't contain enough information, say so.""",
        ),
        (
            "human",
            """Context:
{context}

Question: {question}

Answer:""",
        ),
    ]
)


def generate(state: CRAGState, llm: BaseChatModel) -> CRAGState:
    question = state["question"]
    documents = state["documents"]

    context = "\n\n".join([doc.page_content for doc in documents])

    chain = GENERATOR_PROMPT | llm
    response = chain.invoke({"question": question, "context": context})

    return {
        **state,
        "generation": strip_think_tags(response.content),
    }
