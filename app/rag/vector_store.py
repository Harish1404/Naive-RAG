import logging
from pymongo import UpdateOne
from app.db.mongodb import get_vector_collection

logger = logging.getLogger("uvicorn")


class VectorStore:
    """
    Stores text chunks alongside their embeddings in MongoDB Atlas,
    and performs vector search using Atlas Vector Search ($vectorSearch).
    """

    INDEX_NAME = "vector_index"

    def _get_collection(self):
        return get_vector_collection()

    async def get_existing_ids(self) -> set[str]:
        """Returns all document chunk IDs already present in MongoDB Atlas."""
        try:
            collection = self._get_collection()
            cursor = collection.find({}, {"_id": 1, "id": 1})
            docs = await cursor.to_list(length=50000)
            existing = set()
            for doc in docs:
                if "_id" in doc:
                    existing.add(str(doc["_id"]))
                if "id" in doc:
                    existing.add(str(doc["id"]))
            return existing
        except Exception as e:
            logger.error(f"Failed to fetch existing document IDs from MongoDB: {e}")
            return set()

    async def ensure_vector_index(self):
        """Automatically checks and creates the Atlas Vector Search index if missing."""
        try:
            collection = self._get_collection()
            cursor = collection.list_search_indexes()
            existing_indexes = await cursor.to_list(length=100)
            idx_names = [idx.get("name") for idx in existing_indexes]
            if self.INDEX_NAME not in idx_names:
                from pymongo.operations import SearchIndexModel
                index_model = SearchIndexModel(
                    definition={
                        "fields": [
                            {
                                "type": "vector",
                                "path": "embedding",
                                "numDimensions": 384,
                                "similarity": "cosine"
                            }
                        ]
                    },
                    name=self.INDEX_NAME,
                    type="vectorSearch"
                )
                await collection.create_search_index(model=index_model)
                logger.info(f"✅ Automatically created Atlas Vector Search index '{self.INDEX_NAME}' on MongoDB Atlas.")
        except Exception as e:
            logger.debug(f"Atlas Search index check note: {e}")

    async def add(self, ids: list[str], embeddings: list[list[float]], documents: list[str], metadatas: list[dict]):
        """Stores or updates a batch of document chunks in MongoDB Atlas."""
        if not ids:
            return

        collection = self._get_collection()
        operations = []

        for chunk_id, embedding, doc, meta in zip(ids, embeddings, documents, metadatas):
            doc_body = {
                "_id": chunk_id,
                "id": chunk_id,
                "text": doc,
                "embedding": embedding,
                "source": meta.get("source", "unknown"),
                "metadata": meta,
            }
            operations.append(
                UpdateOne({"_id": chunk_id}, {"$set": doc_body}, upsert=True)
            )

        if operations:
            result = await collection.bulk_write(operations)
            logger.info(f"MongoDB Vector Store updated: {result.upserted_count} inserted, {result.modified_count} modified.")
            await self.ensure_vector_index()


    async def query(self, query_embedding: list[float], top_k: int = 4) -> list[dict]:
        """Finds the `top_k` chunks most similar to the query embedding using MongoDB Atlas Vector Search."""
        collection = self._get_collection()

        pipeline = [
            {
                "$vectorSearch": {
                    "index": self.INDEX_NAME,
                    "path": "embedding",
                    "queryVector": query_embedding,
                    "numCandidates": max(top_k * 10, 50),
                    "limit": top_k,
                }
            },
            {
                "$project": {
                    "text": 1,
                    "source": 1,
                    "score": {"$meta": "vectorSearchScore"},
                }
            }
        ]

        try:
            cursor = collection.aggregate(pipeline)
            results = await cursor.to_list(length=top_k)
            matches = []
            for doc in results:
                matches.append({
                    "text": doc.get("text", ""),
                    "source": doc.get("source", "unknown"),
                    "score": doc.get("score", 0.0),
                    "distance": 1.0 - doc.get("score", 0.0),
                })
            return matches
        except Exception as e:
            logger.warning(
                f"MongoDB Vector Search query failed or 'vector_index' is not configured yet in Atlas UI: {e}"
            )
            return []

