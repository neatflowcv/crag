import chromadb
from langchain_core.documents import Document

from crag.config.settings import settings
from crag.vectorstore.embeddings import get_embedding_model


class VectorStore:
    def __init__(self) -> None:
        self._client = chromadb.PersistentClient(path=str(settings.chroma_persist_dir))
        self._collection = self._client.get_or_create_collection(
            name=settings.chroma_collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._embedding_model = get_embedding_model()

    def add_documents(self, documents: list[Document]) -> None:
        if not documents:
            return

        texts = [doc.page_content for doc in documents]
        embeddings = self._embedding_model.encode(texts).tolist()
        ids = [f"doc_{i}_{hash(text)}" for i, text in enumerate(texts)]
        metadatas = [doc.metadata for doc in documents]

        self._collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

    def search(self, query: str, k: int | None = None) -> list[Document]:
        k = k or settings.retriever_k
        query_embedding = self._embedding_model.encode([query]).tolist()

        results = self._collection.query(
            query_embeddings=query_embedding,
            n_results=k,
        )

        documents = []
        if results["documents"]:
            for i, doc_text in enumerate(results["documents"][0]):
                metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                documents.append(Document(page_content=doc_text, metadata=metadata))

        return documents

    def clear(self) -> None:
        self._client.delete_collection(settings.chroma_collection_name)
        self._collection = self._client.get_or_create_collection(
            name=settings.chroma_collection_name,
            metadata={"hnsw:space": "cosine"},
        )
