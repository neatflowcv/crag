import sys
from pathlib import Path

from langchain_core.documents import Document

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.vectorstore.store import VectorStore


def load_documents_from_directory(directory: Path) -> list[Document]:
    documents = []
    for file_path in directory.glob("*.txt"):
        content = file_path.read_text(encoding="utf-8")
        documents.append(
            Document(
                page_content=content,
                metadata={"source": str(file_path), "filename": file_path.name},
            )
        )
    return documents


def main() -> None:
    documents_dir = Path("docs")

    if not documents_dir.exists():
        print(f"Documents directory not found: {documents_dir}")
        return

    print("Loading documents...")
    documents = load_documents_from_directory(documents_dir)

    if not documents:
        print("No documents found in the directory.")
        return

    print(f"Found {len(documents)} documents.")

    print("Initializing vector store...")
    store = VectorStore()

    print("Clearing existing documents...")
    store.clear()

    print("Adding documents to vector store...")
    store.add_documents(documents)

    print("Document ingestion complete.")


if __name__ == "__main__":
    main()
