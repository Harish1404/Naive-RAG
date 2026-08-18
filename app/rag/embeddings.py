from langchain_google_genai import GoogleGenerativeAIEmbeddings
from app.core.config import settings


class EmbeddingModel:
    """
    Turns text into vectors (lists of numbers) so we can compare how
    similar two pieces of text are by comparing their vectors.

    Uses Google's Gemini embedding API via LangChain.
    """

    def __init__(self, model_name: str = "models/embedding-001"):
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=model_name,
            google_api_key=settings.GEMINI_API_KEY,
        )

    async def embed_texts(self, texts: list[str], batch_size: int = 100) -> list[list[float]]:
        """Embeds many chunks at once — used during ingestion."""
        embeddings_list = []

        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            batch_embeddings = await self.embeddings.aembed_documents(batch)
            embeddings_list.extend(batch_embeddings)

        return embeddings_list

    async def embed_query(self, text: str) -> list[float]:
        """Embeds a single piece of text — used for a user's question."""
        return await self.embeddings.aembed_query(text)

