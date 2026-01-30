from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

from crag.models.state import CRAGState
from crag.utils import strip_think_tags

DECISION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an assistant that decides whether a web search is needed to answer a question.

Given:
- A user question
- Retrieved documents from the local knowledge base
- Whether the documents are relevant (graded by another system)

Decide if a web search is necessary. Consider these factors:
1. If documents are not relevant and the question requires current/real-time information (news, weather, stock prices, recent events), web search is needed.
2. If documents are not relevant but the question is about general knowledge that might not be in search results, web search may not help.
3. If documents are partially relevant but missing key information, web search might help.

Respond in this exact format:
DECISION: yes or no
REASON: [brief explanation in Korean]""",
        ),
        (
            "human",
            """Question: {question}

Documents found: {doc_count}
Documents relevant: {is_relevant}

Document contents:
{documents}

Should we perform a web search? Respond with DECISION and REASON.""",
        ),
    ]
)


def decide_web_search(state: CRAGState, llm: BaseChatModel) -> CRAGState:
    """
    웹 서치가 필요한지 판단하는 노드.

    검색된 문서의 관련성과 질문의 특성을 고려하여
    웹 서치가 필요한지 LLM을 통해 판단한다.
    """
    question = state["question"]
    documents = state["documents"]
    is_relevant = state.get("documents_relevant", False)

    # 문서가 관련 있으면 웹 서치 불필요
    if is_relevant:
        return {
            **state,
            "needs_web_search": False,
            "web_search_reason": "검색된 문서가 질문과 관련이 있어 웹 서치가 필요하지 않습니다.",
        }

    docs_text = "\n\n".join(
        [f"Document {i + 1}:\n{doc.page_content[:500]}..." for i, doc in enumerate(documents)]
    ) if documents else "(No documents retrieved)"

    chain = DECISION_PROMPT | llm
    response = chain.invoke({
        "question": question,
        "doc_count": len(documents),
        "is_relevant": "yes" if is_relevant else "no",
        "documents": docs_text,
    })

    content = strip_think_tags(response.content)

    # 응답 파싱
    needs_search = False
    reason = "판단 결과를 파싱할 수 없습니다."

    lines = content.strip().split("\n")
    for line in lines:
        line_lower = line.lower()
        if line_lower.startswith("decision:"):
            decision = line_lower.replace("decision:", "").strip()
            needs_search = "yes" in decision
        elif line.startswith("REASON:") or line.startswith("reason:"):
            reason = line.split(":", 1)[1].strip() if ":" in line else line

    return {
        **state,
        "needs_web_search": needs_search,
        "web_search_reason": reason,
    }
