import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI

from app.routes.chatbot import router as chatbot_router
from app.rag.rag_pipeline import rag_pipeline
from app.db.mongodb import connect_to_mongo, close_mongo_connection
from app.mcp import client as mcp_client

# Configure logging to output INFO level logs to terminal
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

logger = logging.getLogger("uvicorn.error")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ──
    logger.info("App startup")
    await connect_to_mongo()

    chunk_count = await rag_pipeline.ingest("uploads")
    logger.info(f"RAG startup ingestion complete: {chunk_count} new chunk(s) indexed.")

    await mcp_client.connect()
    logger.info("MCP client connected to GitHub remote server.")

    yield

    # ── Shutdown ──
    logger.info("App shutdown")
    await mcp_client.disconnect()
    await close_mongo_connection()


app = FastAPI(lifespan=lifespan)
app.include_router(chatbot_router)



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
