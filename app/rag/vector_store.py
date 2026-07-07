import chromadb


class VectorStore:
    """
    Stores text chunks alongside their embeddings, and lets us search
    for the chunks whose embeddings are closest to a query embedding.

    Backed by Chroma, saved to disk so it survives server restarts
    (though we currently re-ingest fresh on every startup anyway).
    """

    COLLECTION_NAME = "documents"

    def __init__(self, persist_path: str = "./chroma_db"):
        self.client = chromadb.PersistentClient(path=persist_path)
        self.collection = self.client.get_or_create_collection(self.COLLECTION_NAME)

    def reset_collection(self):
        """Wipes out any previously stored chunks before a fresh ingestion."""
        self.client.delete_collection(self.COLLECTION_NAME)
        self.collection = self.client.get_or_create_collection(self.COLLECTION_NAME)

    def add(self, ids: list[str], embeddings: list[list[float]], documents: list[str], metadatas: list[dict]):
        """Stores a batch of chunks. `metadatas` holds things like {"source": "my_resume.pdf"}."""
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    def query(self, query_embedding: list[float], top_k: int = 4) -> list[dict]:
        """Finds the `top_k` chunks most similar to the query embedding."""
        if self.collection.count() == 0:
            return []

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.collection.count()),
        )

        matches = []
        for text, metadata, distance in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            matches.append({
                "text": text,
                "source": metadata.get("source", "unknown"),
                "distance": distance,
            })

        return matches
