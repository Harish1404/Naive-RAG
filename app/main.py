import logging

from fastapi import FastAPI

from app.routes.chatbot import router as chatbot_router
from app.rag.rag_pipeline import rag_pipeline

logger = logging.getLogger(__name__)

app = FastAPI()
app.include_router(chatbot_router)


@app.on_event("startup")
def ingest_documents():
    chunk_count = rag_pipeline.ingest("uploads")
    logger.info(f"RAG startup ingestion complete: {chunk_count} chunk(s) indexed.")


@app.get("/")
def landing_page():
    return {
        "message": " Hi this is your backend server"
    }

@app.get("/health")
def health():
    return {
        "status": "Your backend server is good"
    }
