from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from src.config.settings import settings
from src.models.state import CRAGState

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


def generate(state: CRAGState) -> CRAGState:
    question = state["question"]
    documents = state["documents"]

    llm = ChatOllama(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
    )

    context = "\n\n".join([doc.page_content for doc in documents])

    chain = GENERATOR_PROMPT | llm
    response = chain.invoke({"question": question, "context": context})

    return {
        **state,
        "generation": response.content,
    }
