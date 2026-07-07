from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    """
    Turns text into vectors (lists of numbers) so we can compare how
    similar two pieces of text are by comparing their vectors.

    Uses a small, local model — no API calls, runs on CPU.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embeds many chunks at once — used during ingestion."""
        return self.model.encode(texts).tolist()

    def embed_query(self, text: str) -> list[float]:
        """Embeds a single piece of text — used for a user's question."""
        return self.model.encode(text).tolist()

