import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.graph.builder import build_graph
from src.models.state import CRAGState


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: uv run python scripts/run_crag.py <question>")
        sys.exit(1)

    question = " ".join(sys.argv[1:])

    print(f"Question: {question}")
    print("-" * 50)

    graph = build_graph()

    initial_state: CRAGState = {
        "question": question,
        "documents": [],
        "web_search_urls": [],
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
