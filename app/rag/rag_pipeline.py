import logging

from app.rag.data_processor import process_folder
from app.rag.embeddings import EmbeddingModel
from app.rag.vector_store import VectorStore

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    Retrieval half of RAG: turns a folder of documents into a searchable
    vector store, and turns a user question into the most relevant chunks.

    Does NOT talk to any LLM — that part lives in ChatService, which asks
    this pipeline for context and then generates the answer.
    """

    def __init__(self):
        self.embedding_model = EmbeddingModel()
        self.vector_store = VectorStore()

    def ingest(self, folder_path: str) -> int:
        """
        Reads every document in `folder_path`, splits it into chunks,
        embeds each chunk, and stores everything in the vector store.

        Returns the number of chunks ingested.
        """
        chunks = process_folder(folder_path)

        if not chunks:
            logger.warning(f"No documents found to ingest in '{folder_path}'.")
            self.vector_store.reset_collection()
            return 0

        texts = [chunk["text"] for chunk in chunks]
        embeddings = self.embedding_model.embed_texts(texts)

        self.vector_store.reset_collection()
        self.vector_store.add(
            ids=[chunk["id"] for chunk in chunks],
            embeddings=embeddings,
            documents=texts,
            metadatas=[{"source": chunk["source"]} for chunk in chunks],
        )

        logger.info(f"Ingested {len(chunks)} chunk(s) from '{folder_path}'.")
        return len(chunks)

    def retrieve(self, query: str, top_k: int = 4) -> list[dict]:
        """Returns the `top_k` chunks most relevant to the given query."""
        query_embedding = self.embedding_model.embed_query(query)
        return self.vector_store.query(query_embedding, top_k=top_k)


# One shared instance — loaded once, reused by every request.
rag_pipeline = RAGPipeline()
