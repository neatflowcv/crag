import logging
import os
from pathlib import Path

import typer

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(message)s",
)

app = typer.Typer(help="CRAG CLI - Corrective RAG 시스템")


@app.command()
def ingest(
    docs_dir: Path = typer.Option(
        Path("docs"),
        "--docs-dir",
        "-d",
        help="문서 디렉토리 경로",
    ),
) -> None:
    """문서를 벡터스토어에 인제스트"""
    from langchain_core.documents import Document

    from crag.vectorstore.store import VectorStore

    if not docs_dir.exists():
        typer.echo(f"Documents directory not found: {docs_dir}")
        raise typer.Exit(1)

    documents = []
    for file_path in docs_dir.glob("*.txt"):
        content = file_path.read_text(encoding="utf-8")
        documents.append(
            Document(
                page_content=content,
                metadata={"source": str(file_path), "filename": file_path.name},
            )
        )

    if not documents:
        typer.echo("No documents found.")
        raise typer.Exit(1)

    typer.echo(f"Found {len(documents)} documents.")
    store = VectorStore()
    store.clear()
    store.add_documents(documents)
    typer.echo("Ingestion complete.")


@app.command()
def run(
    question: str = typer.Argument(..., help="질문"),
) -> None:
    """CRAG로 질문에 답변"""
    import httpx

    from crag.config.settings import settings
    from crag.graph.builder import build_graph

    def check_ollama() -> bool:
        try:
            response = httpx.get(f"{settings.ollama_base_url}/api/tags", timeout=5.0)
            return response.status_code == 200
        except httpx.ConnectError:
            return False

    def get_llm():
        if settings.llm_provider == "openai":
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(model=settings.openai_model)

        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
        )

    if settings.llm_provider == "ollama" and not check_ollama():
        typer.echo(f"Error: Ollama is not running at {settings.ollama_base_url}")
        raise typer.Exit(1)

    typer.echo(f"Question: {question}")
    typer.echo("-" * 50)

    from crag.vectorstore.store import VectorStore

    llm = get_llm()
    store = VectorStore()
    graph = build_graph(llm, store)

    initial_state = {
        "question": question,
        "search_query": question,
        "web_search_queries": [],
        "documents": [],
        "web_search_results": [],
        "generation": "",
        "retry_count": 0,
        "documents_relevant": False,
        "needs_web_search": False,
        "web_search_reason": "",
    }

    result = None
    shown_decision = False

    for event in graph.stream(initial_state):
        for node_name, node_output in event.items():
            result = node_output

            if node_name == "retrieve":
                doc_count = len(node_output.get("documents", []))
                typer.echo(f"\n[Retrieve] {doc_count}개 문서 검색됨")

            elif node_name == "grade_documents":
                is_relevant = node_output.get("documents_relevant", False)
                status = "관련 있음" if is_relevant else "관련 없음"
                typer.echo(f"[Grade] 문서 관련성: {status}")

            elif node_name == "decide_web_search" and not shown_decision:
                needs_search = node_output.get("needs_web_search", False)
                reason = node_output.get("web_search_reason", "")
                decision = "필요함" if needs_search else "불필요"
                typer.echo("\n[Web Search Decision]")
                typer.echo(f"  - 판단: 웹 서치 {decision}")
                typer.echo(f"  - 이유: {reason}")
                if needs_search:
                    typer.echo("\n  -> 웹 서치 진행...")
                else:
                    typer.echo("\n  -> 로컬 문서로 답변 생성...")
                shown_decision = True

            elif node_name == "web_search":
                search_count = len(node_output.get("web_search_results", []))
                typer.echo(f"[Web Search] {search_count}개 결과 수집됨")

    typer.echo(f"\n{'=' * 50}")
    typer.echo(f"Answer:\n{result['generation']}")

    if result.get("retry_count", 0) > 0:
        typer.echo(f"\n(Web search was triggered, retries: {result['retry_count']})")


if __name__ == "__main__":
    app()
