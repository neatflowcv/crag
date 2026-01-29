import sys
from pathlib import Path

import httpx
from langchain_core.language_models import BaseChatModel

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.settings import settings
from src.graph.builder import build_graph
from src.models.state import CRAGState


def check_ollama() -> bool:
    try:
        response = httpx.get(f"{settings.ollama_base_url}/api/tags", timeout=5.0)
        return response.status_code == 200
    except httpx.ConnectError:
        return False


def get_llm() -> BaseChatModel:
    if settings.llm_provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=settings.openai_model)

    from langchain_ollama import ChatOllama

    return ChatOllama(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
    )


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: uv run python scripts/run_crag.py <question>")
        sys.exit(1)

    if settings.llm_provider == "ollama" and not check_ollama():
        print(f"Error: Ollama is not running at {settings.ollama_base_url}")
        print(f"Please start Ollama and ensure {settings.ollama_model} is available.")
        sys.exit(1)

    question = " ".join(sys.argv[1:])

    print(f"Question: {question}")
    print("-" * 50)

    llm = get_llm()
    graph = build_graph(llm)

    initial_state: CRAGState = {
        "question": question,
        "search_query": question,
        "documents": [],
        "web_search_results": [],
        "generation": "",
        "retry_count": 0,
        "documents_relevant": False,
    }

    result = graph.invoke(initial_state)

    print("\nAnswer:")
    print(result["generation"])

    if result.get("retry_count", 0) > 0:
        print(f"\n(Web search was triggered, retries: {result['retry_count']})")


if __name__ == "__main__":
    main()
