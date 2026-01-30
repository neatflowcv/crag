from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

from crag.models.state import CRAGState
from crag.utils import strip_think_tags

GRADER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a grader assessing relevance of retrieved documents to a user question.
If the documents contain information relevant to answering the question, grade them as relevant.
IMPORTANT: The question and documents may be in different languages (e.g., Korean question with English documents).
Assess semantic relevance regardless of language differences.
Give a binary score 'yes' or 'no' to indicate whether the documents are relevant.
Respond with only 'yes' or 'no'.""",
        ),
        (
            "human",
            """Question: {question}

Documents:
{documents}

Are these documents relevant to the question? (yes/no)""",
        ),
    ]
)


def grade_documents(state: CRAGState, llm: BaseChatModel) -> CRAGState:
    question = state["question"]
    documents = state["documents"]

    if not documents:
        return {
            **state,
            "documents_relevant": False,
        }

    docs_text = "\n\n".join(
        [f"Document {i + 1}:\n{doc.page_content}" for i, doc in enumerate(documents)]
    )

    chain = GRADER_PROMPT | llm
    response = chain.invoke({"question": question, "documents": docs_text})

    grade = strip_think_tags(response.content).lower()
    is_relevant = "yes" in grade

    return {
        **state,
        "documents_relevant": is_relevant,
    }
