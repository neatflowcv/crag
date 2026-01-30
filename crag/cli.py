from pathlib import Path

import typer

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

    result = graph.invoke({
        "question": question,
        "search_query": question,
        "documents": [],
        "web_search_results": [],
        "generation": "",
        "retry_count": 0,
        "documents_relevant": False,
    })

    typer.echo(f"\nAnswer:\n{result['generation']}")

    if result.get("retry_count", 0) > 0:
        typer.echo(f"\n(Web search was triggered, retries: {result['retry_count']})")


if __name__ == "__main__":
    app()
